import torch
import torch.onnx
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import onnx
import onnxruntime
from pathlib import Path
import json

class ModelExporter:
    """Export model for deployment."""
    
    def __init__(self,
                 model: torch.nn.Module,
                 input_shape: Dict[str, Tuple],
                 output_dir: Union[str, Path],
                 model_name: str = "e2e_perception"):
        self.model = model
        self.input_shape = input_shape
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_onnx(self,
                    opset_version: int = 12,
                    dynamic_axes: Optional[Dict] = None,
                    simplify: bool = True) -> str:
        """Export model to ONNX format.
        
        Args:
            opset_version: ONNX opset version
            dynamic_axes: Dynamic axes for inputs/outputs
            simplify: Whether to simplify ONNX model
            
        Returns:
            Path to exported model
        """
        # Prepare dummy inputs
        dummy_inputs = {}
        for name, shape in self.input_shape.items():
            dummy_inputs[name] = torch.randn(*shape)
        
        # Set model to eval mode
        self.model.eval()
        
        # Export path
        export_path = self.output_dir / f"{self.model_name}.onnx"
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            dummy_inputs,
            export_path,
            input_names=list(self.input_shape.keys()),
            output_names=['trajectories', 'type_logits'],
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            verbose=False
        )
        
        # Simplify if requested
        if simplify:
            try:
                import onnxsim
                model_onnx = onnx.load(export_path)
                model_simp, check = onnxsim.simplify(model_onnx)
                if check:
                    onnx.save(model_simp, export_path)
            except ImportError:
                print("Warning: onnxsim not installed, skipping simplification")
        
        # Verify exported model
        self._verify_onnx_export(export_path, dummy_inputs)
        
        return str(export_path)
    
    def export_torchscript(self,
                          method: str = 'trace',
                          optimize: bool = True) -> str:
        """Export model to TorchScript format.
        
        Args:
            method: 'trace' or 'script'
            optimize: Whether to optimize the model
            
        Returns:
            Path to exported model
        """
        # Prepare dummy inputs
        dummy_inputs = {}
        for name, shape in self.input_shape.items():
            dummy_inputs[name] = torch.randn(*shape)
        
        # Set model to eval mode
        self.model.eval()
        
        # Export path
        export_path = self.output_dir / f"{self.model_name}.pt"
        
        # Export to TorchScript
        if method == 'trace':
            traced_model = torch.jit.trace(self.model, dummy_inputs)
        else:  # script
            scripted_model = torch.jit.script(self.model)
        
        # Optimize if requested
        if optimize:
            if method == 'trace':
                traced_model = torch.jit.optimize_for_inference(traced_model)
            else:
                scripted_model = torch.jit.optimize_for_inference(scripted_model)
        
        # Save model
        if method == 'trace':
            traced_model.save(export_path)
        else:
            scripted_model.save(export_path)
        
        return str(export_path)
    
    def export_tensorrt(self,
                       precision: str = 'fp32',
                       workspace_size: int = 1 << 30,
                       min_batch_size: int = 1,
                       max_batch_size: int = 16) -> str:
        """Export model to TensorRT format.
        
        Args:
            precision: 'fp32', 'fp16', or 'int8'
            workspace_size: Maximum workspace size in bytes
            min_batch_size: Minimum batch size
            max_batch_size: Maximum batch size
            
        Returns:
            Path to exported model
        """
        try:
            import tensorrt as trt
        except ImportError:
            raise ImportError("TensorRT not installed")
        
        # First export to ONNX
        onnx_path = self.export_onnx(simplify=True)
        
        # Export path
        export_path = self.output_dir / f"{self.model_name}_{precision}.engine"
        
        # Create TensorRT builder and network
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        
        # Parse ONNX file
        parser = trt.OnnxParser(network, logger)
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                raise RuntimeError("Failed to parse ONNX file")
        
        # Configure builder
        config = builder.create_builder_config()
        config.max_workspace_size = workspace_size
        
        if precision == 'fp16':
            if not builder.platform_has_fast_fp16:
                print("Warning: Platform doesn't support fast FP16")
            config.set_flag(trt.BuilderFlag.FP16)
        elif precision == 'int8':
            if not builder.platform_has_fast_int8:
                print("Warning: Platform doesn't support fast INT8")
            config.set_flag(trt.BuilderFlag.INT8)
            # Note: INT8 calibration would be needed here
        
        # Set optimization profiles
        profile = builder.create_optimization_profile()
        for name, shape in self.input_shape.items():
            min_shape = list(shape)
            min_shape[0] = min_batch_size
            max_shape = list(shape)
            max_shape[0] = max_batch_size
            opt_shape = list(shape)
            opt_shape[0] = (min_batch_size + max_batch_size) // 2
            
            profile.set_shape(
                name,
                min=min_shape,
                opt=opt_shape,
                max=max_shape
            )
        config.add_optimization_profile(profile)
        
        # Build engine
        engine = builder.build_engine(network, config)
        if engine is None:
            raise RuntimeError("Failed to build TensorRT engine")
        
        # Save engine
        with open(export_path, 'wb') as f:
            f.write(engine.serialize())
        
        return str(export_path)
    
    def quantize_model(self,
                      calibration_data: Dict[str, torch.Tensor],
                      method: str = 'dynamic',
                      backend: str = 'qnnpack') -> torch.nn.Module:
        """Quantize model for reduced size and faster inference.
        
        Args:
            calibration_data: Data for calibration
            method: 'dynamic' or 'static'
            backend: Quantization backend
            
        Returns:
            Quantized model
        """
        # Set model to eval mode
        self.model.eval()
        
        # Configure quantization
        if method == 'dynamic':
            # Dynamic quantization
            quantized_model = torch.quantization.quantize_dynamic(
                self.model,
                {torch.nn.Linear, torch.nn.Conv2d},
                dtype=torch.qint8
            )
        else:  # static
            # Set quantization backend
            torch.backends.quantized.engine = backend
            
            # Prepare model for static quantization
            qconfig = torch.quantization.get_default_qconfig(backend)
            qconfig_dict = {
                '': qconfig,
                'object_type': [
                    (torch.nn.Linear, qconfig),
                    (torch.nn.Conv2d, qconfig)
                ]
            }
            
            prepared_model = torch.quantization.prepare_qat(
                self.model,
                qconfig_dict
            )
            
            # Calibrate with data
            with torch.no_grad():
                prepared_model(calibration_data)
            
            # Convert to quantized model
            quantized_model = torch.quantization.convert(prepared_model)
        
        return quantized_model
    
    def _verify_onnx_export(self,
                           onnx_path: Union[str, Path],
                           test_inputs: Dict[str, torch.Tensor]):
        """Verify ONNX export by comparing with PyTorch model."""
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        # Create ONNX Runtime session
        ort_session = onnxruntime.InferenceSession(onnx_path)
        
        # Run inference with PyTorch model
        self.model.eval()
        with torch.no_grad():
            torch_outputs = self.model(test_inputs)
        
        # Prepare inputs for ONNX Runtime
        ort_inputs = {
            name: tensor.numpy()
            for name, tensor in test_inputs.items()
        }
        
        # Run inference with ONNX Runtime
        ort_outputs = ort_session.run(None, ort_inputs)
        
        # Compare outputs
        torch_outputs = [out.numpy() for out in torch_outputs]
        
        for torch_out, ort_out in zip(torch_outputs, ort_outputs):
            np.testing.assert_allclose(
                torch_out, ort_out,
                rtol=1e-3, atol=1e-5
            )
        
        print("ONNX export verification successful") 
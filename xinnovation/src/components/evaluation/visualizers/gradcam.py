import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from xinnovation.src.core.registry import VISUALIZERS

@VISUALIZERS.register_module()
class GradCAM:
    """GradCAM visualization for CNN models.
    
    Args:
        model: PyTorch model
        target_layer: Target layer for visualization
    """
    
    def __init__(
        self,
        model,
        target_layer
    ):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
            
        # Register hooks to target layer
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
        
    def generate(self, input_tensor, target_class=None):
        """Generate GradCAM visualization.
        
        Args:
            input_tensor (torch.Tensor): Input tensor
            target_class (int, optional): Target class index
            
        Returns:
            numpy.ndarray: GradCAM heatmap
        """
        # Forward pass
        output = self.model(input_tensor)
        
        # If no target class specified, use the predicted class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Get gradients and activations
        gradients = self.gradients.cpu().numpy()
        activations = self.activations.cpu().numpy()
        
        # Compute weights
        weights = np.mean(gradients, axis=(2, 3))
        
        # Generate heatmap
        heatmap = np.zeros(activations.shape[2:])
        for i, w in enumerate(weights[0]):
            heatmap += w * activations[0, i]
            
        # Normalize heatmap
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / heatmap.max()
        
        return heatmap
        
    def visualize(self, input_tensor, target_class=None, save_path=None):
        """Visualize GradCAM results.
        
        Args:
            input_tensor (torch.Tensor): Input tensor
            target_class (int, optional): Target class index
            save_path (str, optional): Path to save the visualization
        """
        # Generate heatmap
        heatmap = self.generate(input_tensor, target_class)
        
        # Convert input tensor to image
        img = input_tensor[0].cpu().numpy().transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())
        
        # Create figure
        plt.figure(figsize=(10, 5))
        
        # Plot original image
        plt.subplot(121)
        plt.imshow(img)
        plt.title('Original Image')
        plt.axis('off')
        
        # Plot heatmap
        plt.subplot(122)
        plt.imshow(img)
        plt.imshow(heatmap, alpha=0.5, cmap='jet')
        plt.title('GradCAM')
        plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.close() 
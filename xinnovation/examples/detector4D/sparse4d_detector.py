import torch
import torch.nn as nn
from typing import Dict, Optional, Union, Any, List
from xinnovation.src.core import SourceCameraId, DETECTORS, ATTENTION, ANCHOR_GENERATOR, IMAGE_FEATURE_EXTRACTOR, FEEDFORWARD_NETWORK, NORM_LAYERS, build_from_cfg, PLUGINS
from xinnovation.src.components.lightning_module.detectors import FPNImageFeatureExtractor, Anchor3DGenerator, DecoupledMultiHeadAttention
from .mts_feature_aggregator import MultiviewTemporalSpatialFeatureAggregator
from .sparse4d_plugins import AnchorEncoder, TrajectoryRefiner
from xinnovation.src.utils.debug_utils import check_nan_or_inf

__all__ = ["Sparse4DDetector"]

check_abnormal = False

@DETECTORS.register_module()
class Sparse4DDetector(nn.Module):
    def __init__(self, anchor_encoder: Dict,
                    camera_groups: Dict[str, List[SourceCameraId]],
                    decoder_op_orders: List[List[str]], # each list represent a single layer
                    feature_extractors: Dict, 
                    mts_feature_aggregator: Dict,
                    self_attention: Dict,
                    ffn: Dict,
                    refine: Dict,
                    temp_attention: Dict,
                    use_checkpoint: bool = False,
                    **kwargs):
        super().__init__()
        self.camera_groups = camera_groups
        self.decoder_op_orders = decoder_op_orders
        self.use_checkpoint = use_checkpoint
        
        self.anchor_encoder = PLUGINS.build(anchor_encoder)
        self.register_buffer("init_trajs", self.anchor_encoder.get_init_trajs(speed=23.0))
        self.num_queries = self.anchor_encoder.num_queries
        self.query_dim = self.anchor_encoder.embed_dim
        self.query_embed = nn.Parameter(
                torch.zeros([self.anchor_encoder.num_queries, self.query_dim]),
                requires_grad=True,
        )
        
        # Create feature extractors for each camera group
        self.feature_extractors = nn.ModuleDict({
            name: IMAGE_FEATURE_EXTRACTOR.build(cfg)
            for name, cfg in feature_extractors.items()
        })
        
        # build decoder layers
        self.mts_feature_aggregator = build_from_cfg(mts_feature_aggregator, ATTENTION)

        self.decoder_layer_config_map = {
            "temp_attention": [temp_attention, ATTENTION],
            "self_attention": [self_attention, ATTENTION],
            "ffn": [ffn, FEEDFORWARD_NETWORK],
            "refine": [refine, PLUGINS]
        }

        self.decoder_layers = nn.ModuleList()
        for layer_ops in decoder_op_orders:
            layer = nn.ModuleList()
            for op in layer_ops:
                if op == "mts_feature_aggregator":
                    layer.append(self.mts_feature_aggregator)
                    # use the shared mts_feature_aggregator for all layers
                else:
                    cfg, registry = self.decoder_layer_config_map[op]
                    layer.append(build_from_cfg(cfg, registry))
            self.decoder_layers.append(layer)
            
        # Initialize weights
        self.init_weights()
        
    @property
    def use_log_dimension(self) -> bool:
        return self.mts_feature_aggregator.use_log_dimension

    @property
    def sequence_length(self) -> int:
        """Get the sequence length of the detector."""
        return self.mts_feature_aggregator.sequence_length

    def get_init_trajs(self, B: int) -> torch.Tensor:
        return self.init_trajs.repeat(B, 1, 1)

    def init_weights(self):
        # Initialize weights for anchor encoder
        self.anchor_encoder.init_weights()
        # Initialize weights for query embed
        if self.query_embed.requires_grad:
            nn.init.xavier_uniform_(self.query_embed.data, gain=1)
        # Initialize weights for feature extractors
        for extractor in self.feature_extractors.values():
            extractor.init_weights()
        # Initialize weights for decoder layers
        for layer in self.decoder_layers:
            for op in layer:
                if isinstance(op, nn.Module):
                    op.init_weights()
    
    def _extract_features(self, group_name: str, imgs: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract features from images using the specified feature extractor.
        
        Args:
            name: str, name of the feature extractor
            imgs: torch.Tensor, images with shape [B, T, N_cams, C, H, W]
            
        Returns:
            List[torch.Tensor], features with shape [B, T, N_cams, out_channels, H // down_scale_i, W // down_scale_i] for i in range(len(down_scales))
        """
        B, T, N_cams, C, H, W = imgs.shape
        imgs = imgs.view(B * T * N_cams, C, H, W)
        extractor = self.feature_extractors[group_name]
        feats, down_scales = extractor(imgs)
        feats = [feat.view(B, T, N_cams, extractor.out_channels(), H // ds, W // ds) for feat, ds in zip(feats, down_scales)]
        return feats
    
    def forward(self, batch: Dict) -> List[Dict[str, torch.Tensor]]:
        """Forward pass.
        
        Args:
            batch: Dictionary containing:
                - images: Dict[str -> Tensor[B, T, N_cams, C, H, W]], group_name -> img tensor
                - calibrations: Dict[camera_id -> Tensor[B, CameraParamIndex.END_OF_INDEX]]
                - ego_states: Tensor[B, T, EgoStateIndex.END_OF_INDEX]
                - camera_ids: List[SourceCameraId], camera ids
            
        Returns:
            List[torch.Tensor], trajs at each layer
            List[torch.Tensor], quality at each layer
        """
        
        # Extract features from each camera
        all_features_dict: Dict[SourceCameraId, List[torch.Tensor]] = {}  
        for group_name, imgs in batch['images'].items():
            B, T, N_cams = imgs.shape[:3]
            imgs = imgs.view(B * T * N_cams, *imgs.shape[3:])

            extractor = self.feature_extractors[group_name]
            if self.use_checkpoint:
                feats, down_scales = torch.utils.checkpoint.checkpoint(extractor, imgs, use_reentrant=False)
            else:
                feats, down_scales = extractor(imgs)
                # feats: List[Tensor[B*T*N_cams, C, H // down_scale_i, W // down_scale_i]]
            check_nan_or_inf(feats, active=check_abnormal, name="feats")
            for idx in range(len(feats)):
                C_i, H_i, W_i = feats[idx].shape[1:]
                feats[idx] = feats[idx].view(B * T, N_cams, C_i, H_i, W_i)

            for idx, camera_id in enumerate(self.camera_groups[group_name]):
                all_features_dict[camera_id] = [feat[:, idx] for feat in feats] # (B * T, C, H // down_scale_i, W // down_scale_i)
        
        # Decoder part
        init_trajs = self.init_trajs.repeat(B, 1, 1)
        check_nan_or_inf(init_trajs, active=check_abnormal, name="init_trajs")

        tgts = self.query_embed.repeat(B, 1, 1)
        pos_embeds = self.anchor_encoder(init_trajs)
        
        trajs, quality = self._decode_trajs(batch, init_trajs, pos_embeds, tgts, all_features_dict, enable_self_attn=True)
        ct_trajs, ct_quality = self._decode_trajs(batch, init_trajs, pos_embeds, tgts, all_features_dict, enable_self_attn=False)  # cross-attention branch of DAC-DETR
        
        # Clear intermediate tensors to save memory
        del all_features_dict
        torch.cuda.empty_cache()
        
        return trajs, quality, ct_trajs, ct_quality
    
    def _decode_trajs(self, 
                      batch: Dict,
                      trajs: torch.Tensor,
                      trajs_pos_embeds: torch.Tensor,
                      tgts: torch.Tensor, 
                      all_features_dict: Dict[SourceCameraId, List[torch.Tensor]], 
                      enable_self_attn: bool) -> List[Any]:
        """
        Decode the trajectories from the transformer decoder.
        
        Args:
            batch: Batch data,
            trajs: Trajectories,
            trajs_pos_embeds: Positional embeddings of trajectories,
            tgts: Target embeddings,
            all_features_dict: Features from all cameras,
            enable_self_attn: Whether to enable self-attention,
            
        Returns:
            predictions: List[torch.Tensor], predictions of trajectories
            quality_predictions: List[torch.Tensor], predictions of quality
        """
        
        predictions = []
        quality_predictions = []
        tgts = self.query_embed.repeat(trajs.shape[0], 1, 1)
        
        for layer_idx in range(len(self.decoder_op_orders)):
            layer_ops = self.decoder_op_orders[layer_idx]
            layers = self.decoder_layers[layer_idx]
            
            for op_idx in range(len(layer_ops)):
                op, layer = layer_ops[op_idx], layers[op_idx]
                if op == "mts_feature_aggregator":
                    pixels, tgts = self.mts_feature_aggregator(trajs, 
                                                       batch['camera_ids'], 
                                                       tgts, 
                                                       all_features_dict, 
                                                       batch['calibrations'], 
                                                       batch['ego_states'], 
                                                       trajs_pos_embeds)
                    check_nan_or_inf(pixels, active=check_abnormal, name="pixels")
                    check_nan_or_inf(tgts, active=check_abnormal, name="tgts")
                elif op == "temp_attention":
                    raise NotImplementedError("Temp attention is not implemented")
                    # tgt = layer(tgts, pos_embeds, histories_tgts, histories_pos_embeds)
                elif op == "self_attention" and enable_self_attn:
                    tgts = layer(tgts, trajs_pos_embeds)
                    check_nan_or_inf(tgts, active=check_abnormal, name="tgts")
                elif op == "ffn":
                    tgts = layer(tgts)
                    check_nan_or_inf(tgts, active=check_abnormal, name="tgts")
                elif op == "refine":
                    trajs, quality = layer(tgts, trajs_pos_embeds, trajs)
                    check_nan_or_inf(trajs, active=check_abnormal, name="trajs")
                    assert trajs is not None
                    predictions.append(trajs)
                    quality_predictions.append(quality)
            # order: temp_attn => self_attn => mts_feature_aggregator => ffn => refine
        return predictions, quality_predictions
import torch
import torch.nn as nn
from typing import Dict, Optional, Union, Any, List
from xinnovation.src.core import SourceCameraId, DETECTORS, ATTENTION, ANCHOR_GENERATOR, IMAGE_FEATURE_EXTRACTOR, FEEDFORWARD_NETWORK, NORM_LAYERS, build_from_cfg, PLUGINS
from xinnovation.src.components.lightning_module.detectors import FPNImageFeatureExtractor, Anchor3DGenerator, DecoupledMultiHeadAttention
from .mts_feature_aggregator import MultiviewTemporalSpatialFeatureAggregator
from .sparse4d_plugins import AnchorEncoder, TrajectoryRefiner

__all__ = ["Sparse4DDetector"]

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
        for idx, layer_ops in enumerate(decoder_op_orders):
            layer = nn.ModuleList()
            if idx == 0:
                mts_idx = layer_ops.index("mts_feature_aggregator")
                layer_ops = layer_ops[mts_idx:] # start from mts_feature_aggregator
            for op in layer_ops:
                if op == "mts_feature_aggregator":
                    layer.append(self.mts_feature_aggregator)
                    # use the shared mts_feature_aggregator for all layers
                else:
                    cfg, registry = self.decoder_layer_config_map[op]
                    if cfg is None:
                        continue
                    layer.append(build_from_cfg(cfg, registry))
            self.decoder_layers.append(layer)
    
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
        feats = extractor(imgs)
        feats = [feat.view(B, T, N_cams, extractor.out_channel, H // ds, W // ds) for feat, ds in zip(feats, extractor.fpn_downsample_scales)]
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
            if self.use_checkpoint:
                feats = torch.utils.checkpoint.checkpoint(self._extract_features, group_name, imgs, use_reentrant=False)
            else:
                feats = self._extract_features(group_name, imgs) 
                # feats: List[Tensor[B*T*N_cams, C, H // down_scale_i, W // down_scale_i]]
            for i in range(len(feats)):
                C_i, H_i, W_i = feats[i].shape[2:]
                feats[i] = feats[i].view(B * T, N_cams, C_i, H_i, W_i)
            for idx, camera_id in enumerate(self.camera_groups[group_name]):
                all_features_dict[camera_id] = [feat[:, :, idx] for feat in feats] # (B * T, C, H // down_scale_i, W // down_scale_i)
        
        # Decoder part
        trajs_prediction = []
        quality_prediction = []

        tgts = torch.zeros(B, self.num_queries, self.query_dim)
        tgts = tgts.to(self.init_trajs.device)
        pos_embeds = self.anchor_encoder(self.init_trajs).unsqueeze(0).repeat(B, 1, 1)
        
        for layer_idx in range(len(self.decoder_op_orders)):
            layer_ops = self.decoder_op_orders[layer_idx]
            layers = self.decoder_layers[layer_idx]
            
            for op_idx in range(len(layer_ops)):
                op, layer = layer_ops[op_idx], layers[op_idx]
                if op == "mts_feature_aggregator":
                    all_features_dict = self.mts_feature_aggregator(trajs, 
                                                                    batch['camera_ids'], 
                                                                    tgts, 
                                                                    all_features_dict, 
                                                                    batch['calibrations'], 
                                                                    batch['ego_states'], 
                                                                    pos_embeds)
                elif op == "temp_attention":
                    raise NotImplementedError("Temp attention is not implemented")
                    # tgt = layer(tgts, pos_embeds, histories_tgts, histories_pos_embeds)
                elif op == "self_attention":
                    tgts = layer(tgts, pos_embeds)
                elif op == "ffn":
                    tgts = layer(tgts)
                elif op == "refine":
                    trajs, quality = layer(tgts, pos_embeds, trajs)
                    trajs_prediction.append(trajs)
                    quality_prediction.append(quality)
            # order: temp_attn => self_attn => mts_feature_aggregator => ffn => refine
                    
        # Clear intermediate tensors to save memory
        del all_features_dict
        torch.cuda.empty_cache()
        
        return trajs_prediction, quality_prediction
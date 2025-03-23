import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Optional, List, Tuple, Union
import numpy as np
import torch.nn.functional as F
from timm.models import create_model
import os
from base import SourceCameraId, TrajParamIndex, CameraParamIndex, EgoStateIndex, CameraType
from utils.math_utils import generate_bbox_corners_points, get_motion_param_range
from utils.pose_transform import project_points_to_image
from e2e_dataset.dataset import MAX_TRAJ_NB
from .anchor_generator import AnchorGenerator
from configs.config import AnchorEncoderConfig

class TrajectoryDACTransformerLayer(nn.Module):
    """We use DAC-DETR method"""
    
    def __init__(self,  feature_dim: int, num_heads: int = 8, dim_feedforward: int = 2048):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
        
        self.linear1 = nn.Linear(feature_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, feature_dim)
        
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.norm3 = nn.LayerNorm(feature_dim)
        
        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.activation = nn.ReLU()
        
    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
    
    
    def update_query_by_self_attn(self, tgt, 
                                  query_pos: Optional[torch.Tensor] = None,
                                  tgt_mask: Optional[torch.Tensor] = None, 
                                  tgt_key_padding_mask: Optional[torch.Tensor] = None):
        tgt2 = self.self_attn(self.with_pos_embed(tgt, query_pos), 
                              self.with_pos_embed(tgt, query_pos), 
                              value=tgt, 
                              attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        return tgt
    
    def update_query_by_cross_attn(self, tgt, memory,
                                   query_pos: Optional[torch.Tensor] = None,
                                   memory_pos: Optional[torch.Tensor] = None,
                                   memory_mask: Optional[torch.Tensor] = None,
                                   memory_key_padding_mask: Optional[torch.Tensor] = None):
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos), 
                               self.with_pos_embed(memory, memory_pos), 
                               value=tgt, 
                               attn_mask=memory_mask,
                               key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt
    
    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt
    
    def forward_without_self_attn(self, tgt, memory,
                                  query_pos: Optional[torch.Tensor] = None,
                                  memory_pos: Optional[torch.Tensor] = None,
                                  memory_mask: Optional[torch.Tensor] = None,
                                  memory_key_padding_mask: Optional[torch.Tensor] = None):
        tgt = self.update_query_by_cross_attn(tgt, memory, query_pos, memory_pos, memory_mask, memory_key_padding_mask)
        tgt = self.forward_ffn(tgt)
        return tgt
        
    def forward(self, tgt, memory,
                query_pos: Optional[torch.Tensor] = None,
                memory_pos: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None):
        tgt = self.update_query_by_self_attn(tgt, query_pos, tgt_mask, tgt_key_padding_mask)
        tgt = self.update_query_by_cross_attn(tgt, memory, query_pos, memory_pos, memory_mask, memory_key_padding_mask)
        tgt = self.forward_ffn(tgt)
        return tgt


class AnchorEncoder(nn.Module):
    """Encode anchors to features."""
    def __init__(self, anchor_config: Union[Dict, AnchorEncoderConfig]):
        super().__init__()
        if isinstance(anchor_config, AnchorEncoderConfig):
            self.anchor_config = anchor_config
        else:
            self.anchor_config = AnchorEncoderConfig(**anchor_config)
        
        self.position_embed_layer = nn.Sequential(
            nn.Linear(3, self.anchor_config.position_embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(self.anchor_config.position_embedding_dim),
            nn.Linear(self.anchor_config.position_embedding_dim, self.anchor_config.position_embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(self.anchor_config.position_embedding_dim)
        )
        
        self.dimension_embed_layer = nn.Sequential(
            nn.Linear(3, self.anchor_config.dimension_embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(self.anchor_config.dimension_embedding_dim),
            nn.Linear(self.anchor_config.dimension_embedding_dim, self.anchor_config.dimension_embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(self.anchor_config.dimension_embedding_dim)
        )
        
        self.velocity_embed_layer = nn.Sequential(
            nn.Linear(2, self.anchor_config.velocity_embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(self.anchor_config.velocity_embedding_dim),
            nn.Linear(self.anchor_config.velocity_embedding_dim, self.anchor_config.velocity_embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(self.anchor_config.velocity_embedding_dim)
        )

        self._generate_init_trajs()
    
    def _generate_init_trajs(self) -> torch.Tensor:
        self.anchor_generator = AnchorGenerator.create(**self.anchor_config.anchor_generator_config)
        self.anchors = self.anchor_generator.generate_anchors() # [N, 6] x, y, z, length, width, height
        print(f"total anchors: {self.anchors.shape[0]}")
        self.anchor_generator.save_bev_anchor_fig(os.getcwd())
        # self.init_trajs = torch.zeros(self.anchors.shape[0], TrajParamIndex.END_OF_INDEX)
        self.init_trajs = torch.rand(self.anchors.shape[0], TrajParamIndex.END_OF_INDEX)
        self.init_trajs[:, TrajParamIndex.X:TrajParamIndex.Z + 1] = self.anchors[:, :3]
        self.init_trajs[:, TrajParamIndex.LENGTH:TrajParamIndex.HEIGHT + 1] = self.anchors[:, 3:6]
        self.init_trajs[:, TrajParamIndex.X] += 4.5 # front bumper to imu
        return self.init_trajs
    
    def get_init_trajs(self, speed: float = 23.0) -> torch.Tensor:
        self.init_trajs[:, TrajParamIndex.VX] = speed
        return self.init_trajs
    
    def forward(self, trajs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            trajs: Tensor[B, num_queries, TrajParamIndex.END_OF_INDEX]
            
        Returns:
            Tensor[B, num_queries, 256]
        """
        position_embed = self.position_embed_layer(trajs[:, :, TrajParamIndex.X:TrajParamIndex.Z + 1])
        dimension_embed = self.dimension_embed_layer(trajs[:, :, TrajParamIndex.LENGTH:TrajParamIndex.HEIGHT + 1])
        velocity_embed = self.velocity_embed_layer(trajs[:, :, TrajParamIndex.VX:TrajParamIndex.VY + 1])
        return torch.cat([position_embed, dimension_embed, velocity_embed], dim=-1)
        
        
class TrajectoryDecoder(nn.Module):
    """Decode trajectories from features."""
    
    def __init__(self,
                 num_layers: int = 6,
                 feature_dim: int = 256,
                 hidden_dim: int = 512,
                 anchor_encoder_config: AnchorEncoderConfig = None): # Points to sample per face of the unit cube
        super().__init__()

        operation_order = [
            "temp_gnn",
            "gnn",
            "norm",
            "deformable",
            "norm",
            "ffn",
            "norm",
            "refine",
        ] * num_layers
        # delete the 'gnn' and 'norm' layers in the first transformer blocks
        self.operation_order = operation_order[3:]

        self.op_config_map = {
            "temp_gnn": [temp_graph_model, ATTENTION],
            "gnn": [graph_model, ATTENTION],
            "norm": [norm_layer, NORM_LAYERS],
            "ffn": [ffn, FEEDFORWARD_NETWORK],
            "deformable": [deformable_model, ATTENTION],
            "refine": [refine_layer, PLUGIN_LAYERS],
        }
        self.layers = nn.ModuleList(
            [
                build(*self.op_config_map.get(op, [None, None]))
                for op in self.operation_order
            ]
        )

        def build(cfg, registry, default_args):
            if isinstance(cfg, str):
                cfg = registry.get(cfg)
            if isinstance(cfg, dict):
                cfg = registry.get(cfg.pop("type"))(**cfg)
            return cfg

        self.layers= nn.ModuleList(
            [
                build(*self.)
            ]
        
        # Object candidates: Anchors(Positional encoding) + Content queries(Content encoding)
        self.anchor_encoder = AnchorEncoder(anchor_encoder_config)
        self.init_trajs = nn.Parameter(self.anchor_encoder.get_init_trajs(speed=23.0))
        self.query_dim = self.anchor_encoder.anchor_config.position_embedding_dim
        self.num_queries = self.init_trajs.shape[0]
        self.query_contents = nn.Embedding(self.num_queries, self.query_dim)

        # Register unit points
        self.register_buffer('unit_points', generate_bbox_corners_points()) # [3, 9] corners + center
        self.register_buffer('origin_point', torch.zeros(3, 1)) # [3, 1]
        
        # Parameter ranges for normalization: torch.Tensor[TrajParamIndex.HEIGHT + 1, 2]
        ranges = get_motion_param_range()
        self.register_buffer('motion_min_vals', ranges[:, 0])
        self.register_buffer('motion_ranges', ranges[:, 1] - ranges[:, 0])
        
         # Single trajectory parameter head that outputs all trajectory parameters        
        self.motion_head = nn.Sequential(
            nn.Linear(self.query_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, TrajParamIndex.HEIGHT + 1),
            nn.Sigmoid()
        )
        self.cls_head = nn.Sequential(
            nn.Linear(self.query_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, TrajParamIndex.END_OF_INDEX - TrajParamIndex.HEIGHT - 1)
        )
        
        self.feature_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.query_dim)
        )
        
        # Refiner layers
        self.layers = nn.ModuleList([
            TrajectoryDACTransformerLayer(
                feature_dim=self.query_dim,
                num_heads=8,
                dim_feedforward=1024
            ) for _ in range(num_layers)
        ])
        
        # Initialize parameters
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
    def _decode_trajectory(self, x: torch.Tensor) -> torch.Tensor:
        """ Decode trajectory from features."""
        motion_params = self.motion_head(x)
        motion_params = motion_params * self.motion_ranges + self.motion_min_vals
        cls_params = self.cls_head(x)
        traj_params = torch.cat([motion_params, cls_params], dim=-1)
        return traj_params
                
    def _sample_features_by_queries(self, 
                                    queries: torch.Tensor,
                                    features_dict: List[torch.Tensor], 
                                    calibrations: torch.Tensor, 
                                    ego_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            queries: Tensor[B, num_queries, query_dim]
            features_dict: List[Tensor[B, T, C, H, W]] of length N_cams, we don't cat them together 
                           cause we customize the feature map size for each camera
            calibrations: Tensor[B, N_cams, CameraParamIndex.END_OF_INDEX]
            ego_states: Tensor[B, T, EgoStateIndex.END_OF_INDEX]
            
        Returns:
            trajs: Tensor[B, num_queries, TrajParamIndex.END_OF_INDEX]
            features: Tensor[B, num_queries, C, H, W]
            feature_pos: Tensor[B, num_queries, 3]
        """
         
        # predict trajectory parameters to fetch features from features_dict,
        # we don't do cross-attention with all the img features, we only do it with
        # the fetched features by projecting the trajectory in sequential images.
        # It's similar to the way we do in DETR3D. Sparse sampling.
        
        trajs = self._decode_trajectory(queries)
        # pixels, behind_camera = project_points_to_image(trajs, calibrations, ego_states, self.unit_points, normalize=True)
        features, feature_pos = None, None
        # TODO: grid sampling
        return trajs, features, feature_pos
    
        
    def forward(self, 
                features_dict: List[torch.Tensor],
                calibrations: torch.Tensor,
                ego_states: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            features_dict: List[Tensor[B, T, C, H, W]] of length N_cams, we don't cat them together 
                           cause we customize the feature map size for each camera
            calibrations: Tensor[B, N_cams, CameraParamIndex.END_OF_INDEX]
            ego_states: Tensor[B, T, EgoStateIndex.END_OF_INDEX]
            
        Returns:
            List of trajectory parameter tensors [B, num_queries, TrajParamIndex.END_OF_INDEX]
        """
        # Initialize object queries position embedding
        B, T, _ = ego_states.shape
        # Content queries
        content_queries = self.query_contents.weight.unsqueeze(0).repeat(B, 1, 1)
        # Positional queries
        trajs = self.init_trajs.unsqueeze(0).repeat(B, 1, 1)
        pos_queries = self.anchor_encoder(trajs)

        # Deformable Aggregation
        # TODO:


        

        init_trajs, o_features, o_feature_pos = self._sample_features_by_queries(pos_queries, features_dict, calibrations, ego_states)
        c_features, c_feature_pos = o_features, o_feature_pos
        
        o_decoder_outputs, c_decoder_outputs = [init_trajs], [init_trajs] # o: standard decoder, c: cross-attention decoder
    
        o_tgt = torch.zeros(B, self.num_queries, self.query_dim)
        c_tgt = torch.zeros(B, self.num_queries, self.query_dim)
        o_tgt = o_tgt.to(init_trajs.device)
        c_tgt = c_tgt.to(init_trajs.device)
        
        
        c_features = pos_queries2
        for layer in self.layers:
            # 1. update query by transformer layer(standard)
            o_tgt = layer(o_tgt, pos_queries2, pos_queries, o_feature_pos)
            # 2. update query by transformer layer(without self-attention)
            c_tgt = layer.forward_without_self_attn(c_tgt, pos_queries2, pos_queries, c_feature_pos)
            # 3. sample features from features_dict
            o_trajs, o_features, o_feature_pos = self._sample_features_by_queries(o_tgt, features_dict, calibrations, ego_states)
            c_trajs, c_features, c_feature_pos = self._sample_features_by_queries(c_tgt, features_dict, calibrations, ego_states)
            o_decoder_outputs.append(o_trajs)
            c_decoder_outputs.append(c_trajs)
        
            # # 1. update query by transformer layer(standard)
            # o_tgt = layer(o_tgt, o_features, pos_queries, o_feature_pos)
            # # 2. update query by transformer layer(without self-attention)
            # c_tgt = layer.forward_without_self_attn(c_tgt, c_features, pos_queries, c_feature_pos)
            # # 3. sample features from features_dict
            # o_trajs, o_features, o_feature_pos = self._sample_features_by_queries(o_tgt, features_dict, calibrations, ego_states)
            # c_trajs, c_features, c_feature_pos = self._sample_features_by_queries(c_tgt, features_dict, calibrations, ego_states)
            # o_decoder_outputs.append(o_trajs)
            # c_decoder_outputs.append(c_trajs)
            
        return o_decoder_outputs, c_decoder_outputs


class DecoupledSelfAttention(nn.Module):
    """
    Decoupled attention in Sparse4D.
    """
    def __init__(self, feature_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.value_linear = nn.Linear(feature_dim, feature_dim * 2, bias=False)
        self.query_linear = nn.Linear(feature_dim * 2, feature_dim, bias=False)
        self.attn = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)

    def forward(self, query: torch.Tensor, pos_query: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: Tensor[B, N, query_dim]
            pos_query: Tensor[B, N, query_dim]

        Returns:
            Tensor[B, N, query_dim]
        """
        tgt = torch.cat([query, pos_query], dim=1) # [B, 2N, query_dim]
        v = self.value_linear(tgt) # [B, 2N, value_dim]
        
        # self-attention
        tgt = self.attn(tgt, tgt, v)[0] # [B, 2N, value_dim]
        tgt = self.query_linear(tgt) # [B, 2N, query_dim]
        return tgt


class DecoupledCrossAttention(nn.Module):
    """
    Decoupled cross-attention in Sparse4D.
    """
    def __init__(self, feature_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.value_linear = nn.Linear(feature_dim, feature_dim * 2, bias=False)
        self.query_linear = nn.Linear(feature_dim * 2, feature_dim, bias=False)
        self.attn = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)

    def forward(self, tgt: torch.Tensor, pos_tgt: torch.Tensor, memory: torch.Tensor, pos_memory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tgt: Tensor[B, N, query_dim], current query
            pos_tgt: Tensor[B, N, query_dim], current query position encoding
            memory: Tensor[B, M, query_dim], tracked query
            pos_memory: Tensor[B, M, query_dim], tracked query position encoding

        Returns:
            Tensor[B, N, query_dim]
        """

        tgt = torch.cat([tgt, memory], dim=1) # [B, N + M, query_dim]
        pos_tgt = torch.cat([pos_tgt, pos_memory], dim=1) # [B, N + M, query_dim]

        q = torch.cat([tgt, pos_tgt], dim=2) # [B, N + M, 2 * query_dim] 
        k = torch.cat([memory, pos_memory], dim=2) # [B, M, 2 * query_dim]
        v = self.value_linear(memory) # [B, M, 2 * query_dim]

        tgt = self.attn(q, k, v)[0] # [B, N + M, 2 * query_dim]
        tgt = self.query_linear(tgt) # [B, N + M, query_dim]
        return tgt


class TrajectoryDecoderLayer(nn.Module):
    """
    Trajectory Decoder Layer.
    This layer implements the Decoder Layer in Sparse4D.
    It contains 4 parts:
    1. Cross-attention with tracked queries
    2. Self-attention within the updated queries
    3. Deformable Aggregation(sample features from features_dict), which generates new queries
    4. FFN to predict the refined anchor and classification scores
    5. Optional: chose topK anchors and refine them again
    """
    def __init__(self,
                 enable_self_attn: bool = True,
                 enable_cross_attn: bool = True,
                 feature_dim: int = 256,
                 num_heads: int = 8,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1
                 ):
        super().__init__()
        self.enable_self_attn = enable_self_attn
        self.enable_cross_attn = enable_cross_attn
        self.self_attn = DecoupledSelfAttention(feature_dim, num_heads) if enable_self_attn else None
        self.cross_attn = DecoupledCrossAttention(feature_dim, num_heads) if enable_cross_attn else None

    def forward(self, 
                features_dict: List[torch.Tensor],
                calibrations: torch.Tensor,
                ego_states: torch.Tensor,
                trajs: torch.Tensor,
                trajs_embed: torch.Tensor,
                content_queries: torch.Tensor,
                predicted_trajs: Optional[torch.Tensor] = None,
                predicted_trajs_embed: Optional[torch.Tensor] = None,
                predicted_content_queries: Optional[torch.Tensor] = None,
                ) -> List[torch.Tensor]:
        """
        Args:
            features_dict: List[Tensor[B, T, C, H, W]] of length N_cams, we don't cat them together 
                           cause we customize the feature map size for each camera
            calibrations: Tensor[B, N_cams, CameraParamIndex.END_OF_INDEX]
            ego_states: Tensor[B, T, EgoStateIndex.END_OF_INDEX]
            trajs: Tensor[B, num_queries, TrajParamIndex.END_OF_INDEX]
            trajs_embed: Tensor[B, num_queries, query_dim], current query position encoding
            content_queries: Tensor[B, num_queries, query_dim], current query content encoding
            predicted_trajs: Tensor[B, num_queries, TrajParamIndex.END_OF_INDEX], predicted query
            predicted_trajs_embed: Tensor[B, num_queries, query_dim], predicted query position encoding
            predicted_content_queries: Tensor[B, num_queries, query_dim], predicted query content encoding
            
        Returns:
            List of trajectory parameter tensors [B, num_queries, TrajParamIndex.END_OF_INDEX]
        """
        tgt = content_queries
        # 1. Cross-attention with predicted queries
        if self.enable_cross_attn:
            tgt = self.cross_attn(content_queries, trajs_embed, predicted_content_queries, predicted_trajs_embed)
                
        # 2. Self-attention within the updated queries
        if self.enable_self_attn:
            tgt = self.self_attn(tgt, trajs_embed)
            
        # 3. Deformable Aggregation
        pass
        # 4. FFN to predict the refined anchor and classification scores
        pass
        # 5. Optional: chose topK anchors and refine them again
        pass
        

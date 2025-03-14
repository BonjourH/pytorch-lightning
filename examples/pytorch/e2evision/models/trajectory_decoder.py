import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Optional, List, Tuple
import numpy as np
import torch.nn.functional as F
from timm.models import create_model

from base import SourceCameraId, TrajParamIndex, CameraParamIndex, EgoStateIndex, CameraType
from utils.math_utils import generate_bbox_corners_points
from utils.pose_transform import project_points_to_image


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
        self.activation = nn.ReLU()
        
    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
    
    
    def update_query_by_self_attn(self, tgt, 
                                  tgt_mask: Optional[torch.Tensor] = None, 
                                  tgt_key_padding_mask: Optional[torch.Tensor] = None,
                                  query_pos: Optional[torch.Tensor] = None):
        tgt2 = self.self_attn(self.with_pos_embed(tgt, query_pos), 
                              self.with_pos_embed(tgt, query_pos), 
                              value=tgt, 
                              attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        return tgt
    
    def update_query_by_cross_attn(self, tgt, memory,
                                   memory_mask: Optional[torch.Tensor] = None,
                                   memory_key_padding_mask: Optional[torch.Tensor] = None,
                                   query_pos: Optional[torch.Tensor] = None,
                                   memory_pos: Optional[torch.Tensor] = None):
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
        tgt = self.update_query_by_cross_attn(tgt, memory, memory_mask, memory_key_padding_mask, query_pos, memory_pos)
        tgt = self.forward_ffn(tgt)
        return tgt
        
    def forward(self, tgt, memory,
                query_pos: Optional[torch.Tensor] = None,
                memory_pos: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None):
        tgt = self.update_query_by_self_attn(tgt, tgt_mask, tgt_key_padding_mask, query_pos)
        tgt = self.update_query_by_cross_attn(tgt, memory, memory_mask, memory_key_padding_mask, query_pos, memory_pos)
        tgt = self.forward_ffn(tgt)
        return tgt
        
        
class TrajectoryDecoder(nn.Module):
    """Decode trajectories from features."""
    
    def __init__(self,
                 num_layers: int = 6,
                 num_queries: int = 128,
                 feature_dim: int = 256,
                 query_dim: int = 256,
                 hidden_dim: int = 512,
                 num_points: int = 25): # Points to sample per face of the unit cube
        super().__init__()
        self.num_queries = num_queries
        self.query_dim = query_dim
        
        # Object queries
        self.pos_embeddings = nn.Embedding(num_queries, query_dim)
        
        # Register unit points
        self.register_buffer('unit_points', generate_bbox_corners_points()) # [3, 9] corners + center
        self.register_buffer('origin_point', torch.zeros(3, 1)) # [3, 1]
        
        # Parameter ranges for normalization: torch.Tensor[TrajParamIndex.HEIGHT + 1, 2]
        ranges = self._get_motion_param_range()
        self.register_buffer('motion_min_vals', ranges[:, 0])
        self.register_buffer('motion_ranges', ranges[:, 1] - ranges[:, 0])
        
         # Single trajectory parameter head that outputs all trajectory parameters        
        self.motion_head = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, TrajParamIndex.HEIGHT + 1),
            nn.Sigmoid()
        )
        self.cls_head = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, TrajParamIndex.END_OF_INDEX - TrajParamIndex.HEIGHT - 1)
        )
        
        self.feature_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, query_dim)
        )
        
        # Refiner layers
        self.layers = nn.ModuleList([
            TrajectoryDACTransformerLayer(
                feature_dim=query_dim,
                num_heads=8,
                dim_feedforward=1024
            ) for _ in range(num_layers)
        ])
        
        # Initialize parameters
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
    
    def _get_motion_param_range(self)->torch.Tensor:
        """Get parameter ranges for normalization.
        
        Returns:
            Tensor of shape [TrajParamIndex.HEIGHT + 1, 2] containing min/max values
        """
        param_range = torch.zeros(TrajParamIndex.HEIGHT + 1, 2)
        
        # Position ranges (in meters)
        param_range[TrajParamIndex.X] = torch.tensor([-80.0, 250.0])
        param_range[TrajParamIndex.Y] = torch.tensor([-10.0, 10.0])
        param_range[TrajParamIndex.Z] = torch.tensor([-3.0, 5.0])
        
         # Velocity ranges (in m/s)
        param_range[TrajParamIndex.VX] = torch.tensor([-40.0, 40.0])
        param_range[TrajParamIndex.VY] = torch.tensor([-5.0, 5.0])
        
        # Acceleration ranges (in m/s^2)
        param_range[TrajParamIndex.AX] = torch.tensor([-5.0, 5.0])
        param_range[TrajParamIndex.AY] = torch.tensor([-2.0, 2.0])
        
        # Yaw range (in radians)
        param_range[TrajParamIndex.YAW] = torch.tensor([-np.pi, np.pi])
        
        # Dimension ranges (in meters)
        param_range[TrajParamIndex.LENGTH] = torch.tensor([0.2, 25.0])
        param_range[TrajParamIndex.WIDTH] = torch.tensor([0.2, 3.0])
        param_range[TrajParamIndex.HEIGHT] = torch.tensor([0.5, 5.0])
        
        return param_range
    
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
        pixels, behind_camera = project_points_to_image(trajs, calibrations, ego_states, self.unit_points, normalize=True)
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
        pos_queries = self.pos_embeddings.weight.unsqueeze(0).repeat(B, 1, 1)

        init_trajs, o_features, o_feature_pos = self._sample_features_by_queries(pos_queries, features_dict, calibrations, ego_states)
        c_features, c_feature_pos = o_features, o_feature_pos
        
        o_decoder_outputs, c_decoder_outputs = [], [] # o: standard decoder, c: cross-attention decoder
        o_tgt = torch.zeros(B, self.num_queries, self.query_dim)
        c_tgt = torch.zeros(B, self.num_queries, self.query_dim)
        for layer in self.layers:
            # 1. update query by transformer layer(standard)
            o_tgt = layer(o_tgt, o_features, pos_queries, o_feature_pos)
            # 2. update query by transformer layer(without self-attention)
            c_tgt = layer.forward_without_self_attn(c_tgt, c_features, pos_queries, c_feature_pos)
            # 3. sample features from features_dict
            o_trajs, o_features, o_feature_pos = self._sample_features_by_queries(o_tgt, features_dict, calibrations, ego_states)
            c_trajs, c_features, c_feature_pos = self._sample_features_by_queries(c_tgt, features_dict, calibrations, ego_states)
            o_decoder_outputs.append(o_trajs)
            c_decoder_outputs.append(c_trajs)
            
        return init_trajs, o_decoder_outputs, c_decoder_outputs
        
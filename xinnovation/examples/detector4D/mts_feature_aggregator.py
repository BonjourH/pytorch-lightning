import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List, Tuple, Optional, Dict
from xinnovation.src.core import SourceCameraId, TrajParamIndex, CameraParamIndex, EgoStateIndex
from xinnovation.src.core.registry import ATTENTION
from xinnovation.src.utils import generate_bbox_corners_points, project_points_to_image
from xinnovation.src.utils.feature_sampling import grid_sample_fpn_features_parallel_apply, grid_sample_fpn_features
from xinnovation.src.utils.debug_utils import check_nan_or_inf

__all__ = ["MultiviewTemporalSpatialFeatureAggregator"]

check_abnormal = False

@ATTENTION.register_module()
class MultiviewTemporalSpatialFeatureAggregator(nn.Module):
    """Sample features from multiple views and temporal frames.
    
    This class samples features from multiple camera views and temporal frames
    for each anchor point in 3D space and then aggregates them to a single feature.
    """
    
    def __init__(self,
                 query_dim: int = 256,
                 num_learnable_points: int = 8,
                 learnable_points_range: float = 3.0,
                 sequence_length: int = 10,
                 temporal_weight_decay: float = 0.5,
                 camera_nb: int = 7,
                 fpn_levels: int = 3,
                 residual_mode: str = "cat",
                 **kwargs):
        """Initialize the feature sampler.
        
        Args:
            num_points: Number of points to sample for each anchor
            num_temporal_frames: Number of temporal frames to sample
            num_spatial_frames: Number of spatial frames to sample
        """
        super().__init__()
        self.num_learnable_points = num_learnable_points
        self.query_dim = query_dim
        self.learnable_points_range = learnable_points_range
        self.residual_mode = residual_mode
        self.sequence_length = sequence_length
        self.temporal_weight_decay = temporal_weight_decay
        
        # Keypoints configuration
        self.register_buffer('unit_points', generate_bbox_corners_points(with_origin=True)) # [3, 9]
        if num_learnable_points > 0 :
            self.learnable_points = nn.Sequential(
                nn.Linear(query_dim, num_learnable_points * 3),
                nn.Tanh()
            )

        # weight channel
        self.camera_nb = camera_nb
        self.fpn_levels = fpn_levels
        self.kpt_num = self.unit_points.shape[-1] + num_learnable_points
        self.weight_channel =  self.kpt_num * camera_nb * fpn_levels
        self.feature_weights = nn.Sequential(
            nn.Linear(query_dim, self.weight_channel), 
            nn.Sigmoid()
        )
        
        self.init_weights()
                
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
                # nn.init.kaiming_normal_(p)

    def _with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def _generate_learnable_points(self, content_queries: torch.Tensor, pos_queries: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate learnable points.
        
        Args:
            content_queries: Tensor[B, num_queries, query_dim]
            pos_queries: Tensor[B, num_queries, query_dim]
        Returns:
            kpts_offsets: Tensor[B, num_queries, 3, num_learnable_points]
        """
        B, N, _ = content_queries.shape
        queries = self._with_pos_embed(content_queries, pos_queries) # [B, N, query_dim]
        kpts_offsets = self.learnable_points(queries) * self.learnable_points_range
        kpts_offsets = kpts_offsets.view(B, N, 3, -1)
        return kpts_offsets
    
    @torch.no_grad()
    def _generate_temporal_weights(self, ego_states: torch.Tensor) -> torch.Tensor:
        """
        Generate the temporal weights.
        Args:
            ego_states: Tensor[B, T, EgoStateIndex.END_OF_INDEX]
        Returns:
            temporal_weights: Tensor[B, T]
        """
        time_diffs = ego_states[...,EgoStateIndex.TIMESTAMP] - ego_states[:, -1, EgoStateIndex.TIMESTAMP] # [B, T]
        # weight = exp(-time_diffs^2 / temporal_weight_decay)
        temporal_weights = torch.exp(-torch.square(time_diffs) / self.temporal_weight_decay * 3)
        temporal_weights = temporal_weights.to(ego_states.device)
        return temporal_weights
    
    def _generate_kpts_feature_weight(self, trajs: torch.Tensor, content_queries: torch.Tensor, pos_queries: Optional[torch.Tensor] = None, ego_states: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate the weight for the features.
        
        Args:
            trajs: Tensor[B, N, TrajParamIndex.END_OF_INDEX]
            content_queries: Tensor[B, N, query_dim]
            pos_queries: Tensor[B, N, query_dim]
        Returns:
            weights: Tensor[B * T, N, N_cams, P, N_levels]
            NOTE: 我们不学习随着时间变化的权重，可以通过一个weight decay来模拟，比如时间点越越近的点权重越大
        """
        B, N, _ = content_queries.shape
        queries = self._with_pos_embed(content_queries, pos_queries)
        weights = self.feature_weights(queries) # [B, N, weight_channel]
        weights = weights.view(B, N, self.camera_nb, self.kpt_num, self.fpn_levels)
        weights = weights.unsqueeze(1) # [B, 1, N, P, N_cams, N_levels]
        temporal_weights = self._generate_temporal_weights(ego_states)
        weights = weights * temporal_weights.view(B, -1, 1, 1, 1, 1) # [B, T, N, P, N_cams, N_levels]
        weights = weights.view(B * self.sequence_length, N, self.camera_nb, self.kpt_num, self.fpn_levels)
        return weights
    
    def cal_projected_pixels(self,
                             trajs: torch.Tensor,
                             calibrations: torch.Tensor,
                             ego_states: torch.Tensor,
                             content_queries: torch.Tensor,
                             pos_queries: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate the projected pixels and behind camera mask.
        """
        B, N, _ = content_queries.shape
        # 1. Concat the learnable points with the unit points
        if self.num_learnable_points > 0:
            learnable_points = self._generate_learnable_points(content_queries, pos_queries) # [B, N, 3, num_learnable_points]
            unit_points = self.unit_points.unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1) # [B, N, 3, 9]
            kpts_all = torch.cat([unit_points, learnable_points], dim=3) # [B, N, 3, P = 9 + num_learnable_points]
        else:
            kpts_all = self.unit_points # [3, 9]

        check_nan_or_inf(kpts_all, active=check_abnormal, name="kpts_all")

        # 2. Project the points to the image plane
        pixels = project_points_to_image(trajs, calibrations, ego_states, kpts_all, normalize=True)
        check_nan_or_inf(pixels, active=check_abnormal, name="pixels")
        return pixels

    def forward(self,
                trajs: torch.Tensor,
                camera_ids: List[SourceCameraId],
                content_queries: torch.Tensor,
                features_dict: Dict[SourceCameraId, List[torch.Tensor]],
                calibrations: torch.Tensor,
                ego_states: torch.Tensor,
                pos_queries: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.
        
        Args:
            trajs: Tensor[B, N, TrajParamIndex.END_OF_INDEX]
            camera_ids: List[SourceCameraId]
            content_queries: Tensor[B, N, query_dim]
            features_dict: Dict[SourceCameraId, List[Tensor[B*T, C, H, W]]]
            calibrations: Tensor[B, N_cams, CameraParamIndex.END_OF_INDEX]
            ego_states: Tensor[B, T, EgoStateIndex.END_OF_INDEX]
            pos_queries: Tensor[B, N, query_dim]

        Returns:
            pixels: Tensor[B * T, N, N_cams, P, 2]
            new_content_queries: Tensor[B, N, query_dim]
        """
        B, N, _ = trajs.shape
        L, P, N_cams = self.fpn_levels, self.kpt_num, self.camera_nb
        # 1. Project the points to the image plane
        pixels = self.cal_projected_pixels(trajs, calibrations, ego_states, content_queries, pos_queries)
        # pixels: Tensor[B * T, N, N_cams, P, 2]
        invalid_mask = (pixels[..., 0] < 0.0) # [B * T, N, N_cams, P]
        
        # 2. Generate the weights for fusing features
        weights = self._generate_kpts_feature_weight(trajs, content_queries, pos_queries, ego_states)
        # weights: Tensor[B * T, N, N_cams, P, L]
        weights[invalid_mask] = 0.0
        
        # 3. Sample the features
        features_list = []
        
        for ic in range(N_cams):
            camera_id = camera_ids[ic]
            # sample features from different scales
            fpn_features = features_dict[camera_id] # List[Tensor[B*T, C, H_i, W_i]] of different scales
            assert len(fpn_features) == L, f"The number of FPN levels must be equal to the number of feature scales, but got {len(fpn_features)} and {L}"
            pixels_ic = pixels[:, :, ic, :, :] * 2 - 1.0 # [B * T, N, P, 2], convert to [-1, 1] for feature sampling
            check_nan_or_inf(pixels_ic, active=check_abnormal, name=f"pixels_ic_{ic}")
            check_nan_or_inf(fpn_features, active=check_abnormal, name=f"fpn_features_{ic}")
            stacked_sampled_features = grid_sample_fpn_features(fpn_features, 
                                                                pixels_ic,
                                                                mode='bilinear',
                                                                padding_mode='zeros',
                                                                align_corners=True,
                                                                dim=-1) 
            check_nan_or_inf(stacked_sampled_features, active=check_abnormal, name=f"stacked_sampled_features_{ic}")
            features_list.append(stacked_sampled_features) # List[Tensor[B * T, C, N, P, L]]
            
        features = torch.stack(features_list, dim=3) # [B * T, C, N, N_cams, P, L]
        features = features.view(B, self.sequence_length, self.query_dim, N, N_cams, P, L)
        weights = weights.view(B, self.sequence_length, 1, N, N_cams, P, L)
        check_nan_or_inf(features, active=check_abnormal, name="features")
        check_nan_or_inf(weights, active=check_abnormal, name="weights")
        
        # TODO: add dropout for weights: for example drop several temporal frames or some kpts
        weighted_features = features * weights
        weighted_features = weighted_features.sum(dim=(1, 4, 5, 6)) # [B, query_dim, N]
        weights_sum = weights.sum(dim=(1, 4, 5, 6)) # [B, 1, N]
        weights_sum = weights_sum.clamp(min=1e-6)
        check_nan_or_inf(weights_sum, active=check_abnormal, name="weights_sum")
        check_nan_or_inf(weighted_features, active=check_abnormal, name="weighted_features")
        weighted_features = weighted_features / weights_sum # [B, query_dim, N]
        check_nan_or_inf(weighted_features, active=check_abnormal, name="normalized_weighted_features")
        new_content_queries = weighted_features.permute(0, 2, 1) # [B, N, query_dim]

        if self.residual_mode == "cat":
            new_content_queries = torch.cat([content_queries, new_content_queries], dim=-1)
        elif self.residual_mode == "add":
            new_content_queries = content_queries + new_content_queries
        elif self.residual_mode == "none":
            pass
        else:
            raise ValueError(f"Invalid residual mode: {self.residual_mode}")
        
        return pixels, new_content_queries
    
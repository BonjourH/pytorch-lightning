from xinnovation.src.core import PLUGINS, ANCHOR_GENERATOR
from xinnovation.src.components.lightning_module.detectors.plugins import Anchor3DGenerator
import torch
import torch.nn as nn
from xinnovation.src.core.dataclass import TrajParamIndex
from typing import Dict
import os
import math
import numpy as np

__all__ = ["AnchorEncoder", "TrajectoryRefiner"]

def bias_init_with_prob(prior_prob):
    """Initialize the bias according to the given probability value."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init

@PLUGINS.register_module()
class AnchorEncoder(nn.Module):
    """Encode anchors to features."""
    def __init__(self, 
                 pos_embed_dim: int, 
                 dim_embed_dim: int, 
                 orientation_embed_dim: int,
                 vel_embed_dim: int,
                 embed_dim: int,
                 anchor_generator_config: Dict,
                 ):
        super().__init__()
        self.embed_dim = pos_embed_dim + dim_embed_dim + orientation_embed_dim + vel_embed_dim
        assert self.embed_dim == embed_dim, f"embed_dim {self.embed_dim} != {embed_dim}"

        self.anchor_generator = ANCHOR_GENERATOR.build(anchor_generator_config)
        self._generate_init_trajs()
        self.num_queries = self.anchor_generator.get_anchor_num()

        def create_embedding_layer(input_dim: int, output_dim: int):
            return nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.ReLU(),
                nn.LayerNorm(output_dim),
                nn.Linear(output_dim, output_dim),
                nn.ReLU(),
                nn.LayerNorm(output_dim)
            )

        self.position_embed_layer = create_embedding_layer(3, pos_embed_dim)
        self.dimension_embed_layer = create_embedding_layer(3, dim_embed_dim)
        self.orientation_embed_layer = create_embedding_layer(2, orientation_embed_dim)
        self.velocity_embed_layer = create_embedding_layer(2, vel_embed_dim)
        self.fusion_layer = create_embedding_layer(self.embed_dim, self.embed_dim)
        
        self.init_weights()
        
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
                # nn.init.kaiming_normal_(p)

    def _generate_init_trajs(self) -> torch.Tensor:
        self.anchors = self.anchor_generator.get_anchors() # [N, 6] x, y, z, length, width, height
        print(f"total anchors: {self.anchors.shape[0]}")
        self.anchor_generator.save_bev_anchor_fig("/tmp")
        # self.init_trajs = torch.zeros(self.anchors.shape[0], TrajParamIndex.END_OF_INDEX)
        self.init_trajs = torch.zeros(self.anchors.shape[0], TrajParamIndex.END_OF_INDEX)
        self.init_trajs[:, TrajParamIndex.X:TrajParamIndex.Z + 1] = self.anchors[:, :3]
        self.init_trajs[:, TrajParamIndex.LENGTH:TrajParamIndex.HEIGHT + 1] = self.anchors[:, 3:6]
        self.init_trajs[:, TrajParamIndex.X] = self.init_trajs[:, TrajParamIndex.X] + 4.5 # front bumper to imu
        self.init_trajs[:, TrajParamIndex.VX] = 23.0
        self.init_trajs[:, TrajParamIndex.COS_YAW] = 1.0
        self.init_trajs = self.init_trajs.unsqueeze(0)
        return self.init_trajs
    
    def get_init_trajs(self, speed: float = 23.0) -> torch.Tensor:
        self.init_trajs[:, :, TrajParamIndex.VX] = speed
        return self.init_trajs
    
    def forward(self, trajs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            trajs: Tensor[B, N, TrajParamIndex.END_OF_INDEX]
            
        Returns:
            Tensor[B, N, embed_dim]
        """
        assert trajs.dim() == 3, f"trajs.dim() {trajs.dim()} != 3"
        position_embed = self.position_embed_layer(trajs[:, :, TrajParamIndex.X:TrajParamIndex.Z + 1])
        dimension_embed = self.dimension_embed_layer(trajs[:, :, TrajParamIndex.LENGTH:TrajParamIndex.HEIGHT + 1])
        orientation_embed = self.orientation_embed_layer(trajs[:, :, TrajParamIndex.COS_YAW:TrajParamIndex.SIN_YAW + 1])
        velocity_embed = self.velocity_embed_layer(trajs[:, :, TrajParamIndex.VX:TrajParamIndex.VY + 1])
        cat_embed = torch.cat([position_embed, dimension_embed, orientation_embed, velocity_embed], dim=-1)
        return self.fusion_layer(cat_embed)


def linear_relu_ln(output_dim, in_loops, out_loops, input_dim=None):
    if input_dim is None:
        input_dim = output_dim
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU(inplace=True))
            input_dim = output_dim
        layers.append(nn.LayerNorm(output_dim))
    return layers


@PLUGINS.register_module()
class TrajectoryRefiner(nn.Module):
    def __init__(self, query_dim: int,
                       hidden_dim: int,
                       normalize_yaw: bool = False,
                       with_quality_estimation: bool = False):
        super().__init__()
        self.query_dim = query_dim
        self.normalize_yaw = normalize_yaw
        self.with_quality_estimation = with_quality_estimation


        # regression part
        # (X, Y, Z, VX, VY, AX, AY, YAW, COS_YAW, SIN_YAW, LOG(LENGTH), LOG(WIDTH), LOG(HEIGHT))
        self.reg_dim = TrajParamIndex.HEIGHT + 1
        self.reg_layers = nn.Sequential(
            *linear_relu_ln(hidden_dim, 2, 2, query_dim),
            nn.Linear(hidden_dim, self.reg_dim)
        )

        # classification part
        # HAS_OBJECT, STATIC, OCCLUDED, CAR, SUV, LIGHTTRUCK, TRUCK, BUS, PEDESTRIAN, BICYCLE, MOTO, CYCLIST, MOTORCYCLIST, CONE
        self.cls_dim = TrajParamIndex.END_OF_INDEX - self.reg_dim
        self.cls_layers = nn.Sequential(
            *linear_relu_ln(hidden_dim, 1, 2, query_dim),
            nn.Linear(hidden_dim, self.cls_dim),
        )

        # quality estimation part
        self.with_quality_estimation = with_quality_estimation
        if with_quality_estimation:
            self.quality_layers = nn.Sequential(
                *linear_relu_ln(hidden_dim, 1, 2, query_dim),
                nn.Linear(hidden_dim, 2),
            )
            
        self.init_weights()
        
    def init_weights(self):
        """Initialize the weights of the network."""
        for p in self.reg_layers.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
                # nn.init.kaiming_normal_(p)

    def forward(self, content_query: torch.Tensor, 
                      pos_query: torch.Tensor,
                      trajs: torch.Tensor,
                      ) -> torch.Tensor:
        """
        Args:
            content_query: (B, N, query_dim)
            pos_query: (B, N, query_dim)
            trajs: (B, N, TrajParamIndex.END_OF_INDEX)

        Returns:
            refined trajs: (B, N, TrajParamIndex.END_OF_INDEX)
            quality: (B, N, 2)
        """
        feature = content_query + pos_query
        reg_out = self.reg_layers(feature)
        cls_out = self.cls_layers(feature)
        quality_out = self.quality_layers(feature) if self.with_quality_estimation else None

        # Create a new tensor instead of modifying the input
        refined_trajs = trajs.clone()
        
        # 1. Do refinement for trajs regression part
        if self.normalize_yaw:
            reg_out[..., [TrajParamIndex.COS_YAW, TrajParamIndex.SIN_YAW]] = torch.nn.functional.normalize(
                reg_out[..., [TrajParamIndex.COS_YAW, TrajParamIndex.SIN_YAW]], dim=-1
            )
            
        # Update the new tensor instead of the input
        refined_trajs[..., :self.reg_dim] = reg_out + trajs[..., :self.reg_dim]
        
        # TODO: the refined x could varies in a large range, for example we expect the refiner could
        # also refine x from 200 to 100, so it's better to find a way to normalize the x for better training.
        # For example we use the log(dimension) to remove the dimension difference. we want to deal with x
        # in a similar way.
        
        # 2. Set regression part for cls part
        refined_trajs[..., -self.cls_dim:] = cls_out
        
        return refined_trajs, quality_out

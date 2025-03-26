import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

from xinnovation.src.core.registry import HEADS
from xinnovation.src.components.lightning_module.losses import CrossEntropyLoss, L1Loss, GIoULoss
from xinnovation.src.components.lightning_module.transformer import DetrTransformer
from xinnovation.src.components.lightning_module.positional_encoding import SinePositionalEncoding

@HEADS.register_module()
class DETRHead(nn.Module):
    """DETR head for end-to-end object detection.
    
    Args:
        num_classes (int): Number of object classes.
        num_queries (int): Number of object queries.
        in_channels (int): Number of input channels.
        num_encoder_layers (int): Number of encoder layers.
        num_decoder_layers (int): Number of decoder layers.
        dim_feedforward (int): Dimension of feedforward network.
        hidden_dim (int): Hidden dimension of transformer.
        dropout (float): Dropout rate.
        nheads (int): Number of attention heads.
        num_encoder_stages (int): Number of encoder stages.
        num_decoder_stages (int): Number of decoder stages.
        pre_norm (bool): Whether to use pre-norm transformer.
        with_box_refine (bool): Whether to use box refinement.
        as_two_stage (bool): Whether to use two-stage transformer.
        transformer (dict): Transformer configuration.
        positional_encoding (dict): Positional encoding configuration.
        loss_cls (dict): Classification loss configuration.
        loss_bbox (dict): Bounding box loss configuration.
        loss_iou (dict): IoU loss configuration.
    """
    
    def __init__(
        self,
        num_classes: int,
        num_queries: int,
        in_channels: int,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        nheads: int = 8,
        num_encoder_stages: int = 1,
        num_decoder_stages: int = 1,
        pre_norm: bool = False,
        with_box_refine: bool = False,
        as_two_stage: bool = False,
        transformer: Optional[Dict] = None,
        positional_encoding: Optional[Dict] = None,
        loss_cls: Optional[Dict] = None,
        loss_bbox: Optional[Dict] = None,
        loss_iou: Optional[Dict] = None
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.in_channels = in_channels
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.nheads = nheads
        self.num_encoder_stages = num_encoder_stages
        self.num_decoder_stages = num_decoder_stages
        self.pre_norm = pre_norm
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        
        # Transformer
        self.transformer = DetrTransformer(**transformer)
        
        # Positional encoding
        self.positional_encoding = SinePositionalEncoding(**positional_encoding)
        
        # Embedding layers
        self.embed_dims = hidden_dim
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # Classification head
        self.fc_cls = nn.Linear(hidden_dim, num_classes + 1)
        
        # Regression head
        self.reg_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )
        
        # Loss functions
        self.loss_cls = CrossEntropyLoss(**loss_cls)
        self.loss_bbox = L1Loss(**loss_bbox)
        self.loss_iou = GIoULoss(**loss_iou)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights of the head."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
                
    def forward(
        self,
        x: torch.Tensor,
        img_metas: List[Dict],
        gt_bboxes: Optional[List[torch.Tensor]] = None,
        gt_labels: Optional[List[torch.Tensor]] = None,
        gt_bboxes_ignore: Optional[List[torch.Tensor]] = None
    ) -> Dict:
        """Forward function.
        
        Args:
            x (torch.Tensor): Input features.
            img_metas (List[Dict]): Image meta information.
            gt_bboxes (List[torch.Tensor], optional): Ground truth bboxes.
            gt_labels (List[torch.Tensor], optional): Ground truth labels.
            gt_bboxes_ignore (List[torch.Tensor], optional): Ground truth bboxes to be ignored.
            
        Returns:
            Dict: A dictionary containing loss and predictions.
        """
        batch_size = x.size(0)
        
        # Add positional encoding
        pos_embed = self.positional_encoding(x)
        x = x + pos_embed
        
        # Prepare query embeddings
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Transformer forward
        hs = self.transformer(x, query_embed)
        
        # Classification and regression predictions
        outputs_class = self.fc_cls(hs)
        outputs_coord = self.reg_ffn(hs).sigmoid()
        
        # Prepare predictions
        all_cls_scores = outputs_class[-1]
        all_bbox_preds = outputs_coord[-1]
        
        # Prepare targets
        if gt_bboxes is not None and gt_labels is not None:
            targets = self.get_targets(
                all_cls_scores,
                all_bbox_preds,
                gt_bboxes,
                gt_labels,
                img_metas,
                gt_bboxes_ignore
            )
        else:
            targets = None
            
        # Compute losses
        if targets is not None:
            losses = self.loss(
                all_cls_scores,
                all_bbox_preds,
                targets
            )
        else:
            losses = {}
            
        # Prepare outputs
        outputs = {
            'loss': losses,
            'all_cls_scores': all_cls_scores,
            'all_bbox_preds': all_bbox_preds
        }
        
        return outputs
        
    def get_targets(
        self,
        cls_scores: torch.Tensor,
        bbox_preds: torch.Tensor,
        gt_bboxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        img_metas: List[Dict],
        gt_bboxes_ignore: Optional[List[torch.Tensor]] = None
    ) -> Dict:
        """Get targets for loss computation.
        
        Args:
            cls_scores (torch.Tensor): Classification scores.
            bbox_preds (torch.Tensor): Bounding box predictions.
            gt_bboxes (List[torch.Tensor]): Ground truth bboxes.
            gt_labels (List[torch.Tensor]): Ground truth labels.
            img_metas (List[Dict]): Image meta information.
            gt_bboxes_ignore (List[torch.Tensor], optional): Ground truth bboxes to be ignored.
            
        Returns:
            Dict: A dictionary containing target information.
        """
        num_imgs = cls_scores.size(0)
        
        # Prepare targets
        targets = {
            'labels': [],
            'boxes': [],
            'img_indices': []
        }
        
        for img_idx in range(num_imgs):
            # Get ground truth for current image
            gt_bbox = gt_bboxes[img_idx]
            gt_label = gt_labels[img_idx]
            
            # Add to targets
            targets['labels'].append(gt_label)
            targets['boxes'].append(gt_bbox)
            targets['img_indices'].append(torch.full_like(gt_label, img_idx))
            
        # Stack targets
        targets['labels'] = torch.cat(targets['labels'])
        targets['boxes'] = torch.cat(targets['boxes'])
        targets['img_indices'] = torch.cat(targets['img_indices'])
        
        return targets
        
    def loss(
        self,
        cls_scores: torch.Tensor,
        bbox_preds: torch.Tensor,
        targets: Dict
    ) -> Dict:
        """Compute losses.
        
        Args:
            cls_scores (torch.Tensor): Classification scores.
            bbox_preds (torch.Tensor): Bounding box predictions.
            targets (Dict): Target information.
            
        Returns:
            Dict: A dictionary containing loss values.
        """
        # Classification loss
        loss_cls = self.loss_cls(
            cls_scores.reshape(-1, self.num_classes + 1),
            targets['labels']
        )
        
        # Bounding box loss
        loss_bbox = self.loss_bbox(
            bbox_preds,
            targets['boxes']
        )
        
        # IoU loss
        loss_iou = self.loss_iou(
            bbox_preds,
            targets['boxes']
        )
        
        # Combine losses
        losses = {
            'loss_cls': loss_cls,
            'loss_bbox': loss_bbox,
            'loss_iou': loss_iou
        }
        
        return losses 
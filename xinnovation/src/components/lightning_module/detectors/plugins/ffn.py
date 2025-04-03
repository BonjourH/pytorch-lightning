from xinnovation.src.core import FEEDFORWARD_NETWORK, ACTIVATION, DROPOUT, NORM_LAYERS, build_from_cfg
import torch
import torch.nn as nn
from torch.nn import Sequential, Linear

__all__ = ["AsymmetricFFN"]


@FEEDFORWARD_NETWORK.register_module()
class AsymmetricFFN(nn.Module):
    def __init__(
        self,
        in_channels=None,
        pre_norm=None,
        post_norm=None,
        embed_dims=256,
        feedforward_channels=1024,
        num_fcs=2,
        act_cfg=dict(type="ReLU", inplace=True),
        ffn_drop=0.1,
        dropout_layer=None,
        add_identity=True,
        **kwargs
    ):
        """
        Args:
            in_channels: 输入通道数
            pre_norm: 预归一化层
            embed_dims: 输出通道数
            feedforward_channels: 前馈通道数
            num_fcs: 前馈层数
            act_cfg: 激活函数配置
            ffn_drop: 前馈层dropout率
            dropout_layer: 丢弃层配置
            add_identity: 是否添加恒等映射
            **kwargs: 其他参数

            FFN(x) = (W1 * x + b1) + f2(W2 * f1(W3 * x + b3) + b2)
            first_part = W1 * x + b1 is the identity mapping
            second_part = f2(W2 * f1(W3 * x + b3) + b2) is the feedforward network
            f1 and f2 are the activation functions
            W1, W2, W3 are the weights
            b1, b2, b3 are the biases

        """
        super(AsymmetricFFN, self).__init__()
        assert num_fcs >= 2, (
            "num_fcs should be no less " f"than 2. got {num_fcs}."
        )
        self.pre_norm = build_from_cfg(pre_norm, NORM_LAYERS)
        self.post_norm = build_from_cfg(post_norm, NORM_LAYERS)
        
        self.in_channels = in_channels if in_channels is not None else embed_dims
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.activate = ACTIVATION.build(act_cfg)
            
        layers = []
        
        for _ in range(num_fcs - 1):
            layers.append(
                Sequential(
                    Linear(in_channels, feedforward_channels),
                    self.activate,
                    nn.Dropout(ffn_drop),
                )
            )
            in_channels = feedforward_channels
        layers.append(Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = Sequential(*layers)
        self.dropout_layer = (
            nn.Dropout(ffn_drop)
            if dropout_layer
            else torch.nn.Identity()
        )
        self.add_identity = add_identity
        if self.add_identity:
            self.identity_fc = (
                torch.nn.Identity()
                if in_channels == embed_dims
                else Linear(self.in_channels, embed_dims)
            )
        
    def init_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight.data)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias.data)
        if self.add_identity:
            if isinstance(self.identity_fc, nn.Linear):
                nn.init.xavier_normal_(self.identity_fc.weight.data)
                if self.identity_fc.bias is not None:
                    nn.init.zeros_(self.identity_fc.bias.data)
        if self.pre_norm is not None:
            self.pre_norm.init_weights()
        if self.post_norm is not None:
            self.post_norm.init_weights()
        

    def forward(self, x):
        if self.pre_norm is not None:
            x = self.pre_norm(x)
        out = self.layers(x) # inchannels -> feedforward_channels -> embed_dims
        out = self.dropout_layer(out)
        if self.add_identity:
            out = self.identity_fc(x) + out
        if self.post_norm is not None:
            out = self.post_norm(out)
        return out

import torch
import torch.nn as nn
from collections import OrderedDict
import timm 
from timm.models.vision_transformer import Block

from .cross_attention import CrossAttentionBlock
from .fpn import FPN

class StrokeAttentionHead(nn.Module):
    '''
    Cross-Attention-based Head
    '''
    def __init__(self, stroke_num=256, stroke_dim=13, encoder_embed_dim=768, self_attn_depth=4):
        super(StrokeAttentionHead, self).__init__()

        self.stroke_tokens = nn.Parameter(torch.zeros(1, stroke_dim, stroke_num))
        self.cross_attn_block = CrossAttentionBlock(x_dim=encoder_embed_dim, y_dim=stroke_num, num_heads=8)
        self.self_attn_blocks = nn.Sequential(*[
            Block(
                dim=stroke_num, num_heads=8, mlp_ratio=4., qkv_bias=True, drop=0.,
                attn_drop=0., norm_layer=nn.LayerNorm, act_layer=nn.GELU)
            for i in range(self_attn_depth)])
        self.linear_head = nn.Linear(stroke_dim, stroke_dim)
        
    def forward(self, x):
        x = self.cross_attn_block(x, self.stroke_tokens.repeat(x.shape[0],1,1))
        x = self.self_attn_blocks(x)
        x = self.linear_head(x.permute(0, 2, 1))
        return x

class StrokeAttentionPredictorV3SAM(nn.Module):
    '''
    Cross-Attention-based Predictor
    '''
    def __init__(self, stroke_num=512, stroke_dim=13, self_attn_depth=1, in_channels=4):
        super(StrokeAttentionPredictorV3SAM, self).__init__()
        self.feature_extractor = timm.models.vision_transformer.vit_small_patch16_224(in_chans=in_channels)

        self.stroke_head = StrokeAttentionHead(stroke_num=stroke_num, stroke_dim=stroke_dim, encoder_embed_dim=self.feature_extractor.embed_dim, self_attn_depth=self_attn_depth)

    def extract_features(self, x):
        x = self.feature_extractor.patch_embed(x)
        cls_token = self.feature_extractor.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x = self.feature_extractor.pos_drop(x + self.feature_extractor.pos_embed)
        x = self.feature_extractor.blocks(x)
        x = self.feature_extractor.norm(x)
        return x[:,1:]

    def forward(self, x):
        x = self.extract_features(x)
        x = self.stroke_head(x)
        return torch.sigmoid(x)
    

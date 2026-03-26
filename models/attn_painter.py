import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from .encoder import StrokeAttentionPredictorV3SAM
from .render_oil import FCNOil
import cv2
import numpy as np
import random

channel_mean = torch.tensor([0.485, 0.456, 0.406])
channel_std = torch.tensor([0.229, 0.224, 0.225])

MEAN = [-mean/std for mean, std in zip(channel_mean, channel_std)]
STD = [1/std for std in channel_std]

class AttnPainterOilDensity(nn.Module):
    '''
    based on AttnPainterV5
    '''
    def __init__(self, stroke_num=256, stroke_dim=8, width=128):
        super(AttnPainterOilDensity, self).__init__()
        self.encoder = StrokeAttentionPredictorV3SAM(stroke_num=stroke_num, stroke_dim=stroke_dim)
        self.stroke_num = stroke_num
        self.device = 'cuda'
        self.render = FCNOil(5,True,True)


        self.width = width
        for p in self.render.parameters():
            p.requires_grad = False


    def forward(self, img, masks):
        masks = masks.float()
        x = torch.cat([img, masks], dim=1)
        # x = img
        strokes = self.encoder(x).reshape(-1, 8)
        pred,_ = self.rendering(strokes, batch_size=img.shape[0])
        pred.reshape(img.shape[0], -1, pred.shape[-2], pred.shape[-1])

        return pred, TF.normalize(TF.resize(img, (pred.shape[-2], pred.shape[-1])), mean=MEAN, std=STD)
    
    def real_forward(self, img, masks, res=128):
        masks = masks.float()
        x = torch.cat([img, masks], dim=1)
        strokes = self.encoder(x)[:,:,:8].reshape(-1, 8)

        pred = self.real_rendering(strokes, batch_size=img.shape[0], res=res)
        pred.reshape(img.shape[0], -1, pred.shape[-2], pred.shape[-1])

        return pred, TF.normalize(TF.resize(img, (pred.shape[-2], pred.shape[-1])), mean=MEAN, std=STD)

    def rendering(self, strokes, batch_size):

        color_stroke, alpha, edge, _ = self.render(strokes)
        color_stroke = color_stroke.reshape(batch_size, -1, 3, self.width, self.width)
        alpha = alpha.reshape(batch_size, -1, 1, self.width, self.width)
        params = strokes.reshape(batch_size, -1, 8)

        # stroke = stroke.reshape(batch_size, -1, 1, self.width, self.width)
        stroke_num_map = torch.arange(1, self.stroke_num + 1).reshape(1, self.stroke_num, 1, 1, 1).repeat(batch_size, 1, 1, 128, 128).to(self.device)

        stroke_s = (params[:, :, 2] * params[:, :, 3]).reshape(batch_size, -1, 1, 1, 1).repeat(1, 1, 1, 128, 128)*(alpha > 0.1)
        stroke_num_map_draw = stroke_num_map*(alpha > 0.1)
        stroke_num_map_draw_topk, stroke_num_map_draw_topk_indices = stroke_num_map_draw.topk(10, dim=1)

        color_stroke_topk = color_stroke.gather(dim=1, index = stroke_num_map_draw_topk_indices.repeat(1, 1, 3, 1, 1))
        alpha = alpha.gather(dim=1, index = stroke_num_map_draw_topk_indices)
        
        stroke_s_map_topk = stroke_s.gather(dim=1, index = stroke_num_map_draw_topk_indices)

        canvas = torch.ones(batch_size, 3, self.width, self.width).to(self.device)
        den_map = torch.ones(batch_size, 1, self.width, self.width).to(self.device)
        
        for i in range(color_stroke_topk.shape[1]):
            canvas = canvas * (1 - alpha[:, 9 - i]) + alpha[:, 9 - i] * color_stroke_topk[:,  9 - i]
            den_map = den_map * (1 - alpha[:, 9 - i]) + alpha[:, 9 - i] * stroke_s_map_topk[:, 9 - i]
            
        return  canvas, den_map
    
    def density_loss(self, img, density_tensor):
        
        x = torch.cat([img, density_tensor], dim=1)
        strokes = self.encoder(x).reshape(-1, 8)
        pred, density_pred = self.rendering(strokes, batch_size=x.shape[0])
        pred.reshape(x.shape[0], -1, pred.shape[-2], pred.shape[-1])
        density_pred.reshape(x.shape[0], -1, pred.shape[-2], pred.shape[-1])

        density_tensor = TF.resize(density_tensor, (pred.shape[-2], pred.shape[-1]))

        loss_mse = F.mse_loss(pred, TF.normalize(TF.resize(img, (pred.shape[-2], pred.shape[-1])), mean=MEAN, std=STD).detach())
        loss_density = (density_tensor*density_pred).mean()

        loss = loss_mse + 0.05*loss_density

        return loss, density_tensor, loss_mse, 0.05*loss_density
    
    def real_rendering(self, strokes, batch_size, res=128):

        color_stroke, alpha, edge, stroke = self.render.real_forward_2(strokes, res=res)
        color_stroke = color_stroke.reshape(batch_size, -1, 3, res, res)
        alpha = alpha.reshape(batch_size, -1, 1, res, res)

        # make the stroke timestamp
        stroke_num_map = torch.arange(1, self.stroke_num + 1).reshape(1, self.stroke_num, 1, 1, 1).repeat(batch_size, 1, 1, res, res).to(self.device)
        stroke_num_map_draw = stroke_num_map*(alpha > 0.5)
        stroke_num_map_draw_topk, stroke_num_map_draw_topk_indices = stroke_num_map_draw.topk(10, dim=1)
        color_stroke_topk = color_stroke.gather(dim=1, index = stroke_num_map_draw_topk_indices.repeat(1, 1, 3, 1, 1))
        alpha = alpha.gather(dim=1, index = stroke_num_map_draw_topk_indices)
        canvas = torch.ones(batch_size, 3, res, res).to(self.device)
        
        for i in range(color_stroke_topk.shape[1]):
            canvas = canvas * (1 - alpha[:, 9 - i]) + alpha[:, 9 - i]*color_stroke_topk[:,  9 - i]
            
        return  canvas

    def real_forward_3(self, img, masks, res=512, col=1, row=1, overlap=10): # 调整分块笔触参数再统一渲染
        masks = masks.float()
        x = torch.cat([img, masks], dim=1)
        strokes = self.encoder(x)[:,:,:8]
        pred = self.real_rendering_3(strokes, blocks=img.shape[0], res=res, col=col, row=row, overlap=overlap)

        return pred, None

    def real_rendering_3(self, strokes, blocks, res=128, col=1, row=1, overlap=10): # 调整分块笔触参数再统一渲染
        print(blocks)
        strokes = self.merge_stroke_parameters(strokes, col=col, row=row, res=res, overlap=overlap)
        strokes = strokes.permute(1, 0, 2)
        strokes = strokes.reshape(-1, 8)
        batch_size = 1
        canvas = torch.ones(batch_size, 3, res, res).cpu()
        color_stroke, alpha, edge, stroke = self.render.real_forward_2(strokes, res=res)
        color_stroke = color_stroke.reshape(batch_size, -1, 3, res, res).cpu()
        alpha = alpha.reshape(batch_size, -1, 1, res, res).cpu()
        alpha = torch.where(alpha > 0.5, 1.0, 0.0)
        for i in range(color_stroke.shape[1]):
            canvas = canvas * (1 - alpha[:, i]) + alpha[:, i] * color_stroke[:, i]

        return canvas

    def merge_stroke_parameters(self, strokes, col, row, res, overlap):
        pi = torch.acos(torch.tensor(-1.))
        for i in range(row):
            for j in range(col):
                block_x_offset = j * ((res / col - overlap) / res)  # 水平方向偏移
                block_y_offset = i * ((res / row - overlap) / res)  # 垂直方向偏移
                scale_x = (res / col + overlap) / res    # 水平方向缩放因子
                scale_y = (res / row + overlap) / res  # 垂直方向缩放因子
                block_strokes = strokes[i * col + j, :, :] #当前块的参数
                block_strokes[:, 0] *= scale_x
                block_strokes[:, 0] += block_x_offset
                block_strokes[:, 1] *= scale_y
                block_strokes[:, 1] += block_y_offset
                sin_theta = torch.sin(pi * block_strokes[:, 4])
                cos_theta = torch.cos(pi * block_strokes[:, 4])
                w_scale = torch.sqrt((cos_theta * scale_x)**2 + (sin_theta * scale_y)**2)
                h_scale = torch.sqrt((sin_theta * scale_x)**2 + (cos_theta * scale_y)**2)
                block_strokes[:, 2] = torch.where(block_strokes[:, 2] < 0.05, block_strokes[:, 2] * 0.0, block_strokes[:, 2] * w_scale)
                block_strokes[:, 3] = torch.where(block_strokes[:, 3] < 0.05, block_strokes[:, 3] * 0.0, block_strokes[:, 3] * h_scale)
                block_strokes[:, 4] = torch.atan2(scale_y * torch.sin(pi * block_strokes[:, 4]), scale_x * torch.cos(pi * block_strokes[:, 4])) / pi

        return strokes
    

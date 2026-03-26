import os
import cv2
import torch
import numpy as np
import argparse
import torchvision.transforms.functional as TF
from models import attn_painter
from tqdm import tqdm
import time

width = 512 #128
channel_mean = torch.tensor([0.485, 0.456, 0.406])
channel_std = torch.tensor([0.229, 0.224, 0.225])
def small2large(x):
    x = x.reshape(args.row_divide, args.col_divide, width, width, -1)
    x = np.transpose(x, (0, 2, 1, 3, 4))
    x = x.reshape(args.row_divide * width, args.col_divide * width, -1)

    return x

def large2small(x):
    x = x.reshape(args.row_divide, width, args.col_divide, width, 3)
    x = np.transpose(x, (0, 2, 1, 3, 4))
    x = x.reshape(canvas_cnt, width, width, 3)
    return x

def large2small_overlap(x, row_divide, col_divide, overlap=10):
    block_height = width // row_divide + overlap
    block_width = width // col_divide + overlap
    blocks = np.zeros((row_divide * col_divide, block_height, block_width, 3), dtype=x.dtype)
    idx = 0
    for i in range(row_divide):
        for j in range(col_divide):
            start_h = i * (width // row_divide - overlap)
            start_w = j * (width // col_divide - overlap)
            block = x[start_h:start_h + block_height, start_w:start_w + block_width, :]
            blocks[idx] = block
            idx += 1

    return blocks

def smooth(img):
    def smooth_pix(img, tx, ty):
        if tx == args.row_divide * width - 1 or ty == args.col_divide * width - 1 or tx == 0 or ty == 0: 
            return img
        img[tx, ty] = (img[tx, ty] + img[tx + 1, ty] + img[tx, ty + 1] + img[tx - 1, ty] + img[tx, ty - 1] + img[tx + 1, ty - 1] + img[tx - 1, ty + 1] + img[tx - 1, ty - 1] + img[tx + 1, ty + 1]) / 9
        return img

    for p in range(args.row_divide):
        for q in range(args.col_divide):
            x = p * width
            y = q * width
            for k in range(width):
                img = smooth_pix(img, x + k, y + width - 1)
                if q != args.col_divide - 1:
                    img = smooth_pix(img, x + k, y + width)
            for k in range(width):
                img = smooth_pix(img, x + width - 1, y + k)
                if p != args.row_divide - 1:
                    img = smooth_pix(img, x + width, y + k)
    return img

def save_img(res, img_name, divide=False, output_dir='output'):
    output = res.detach().cpu().numpy() # d * d, 3, width, width    
    output = np.transpose(output, (0, 2, 3, 1))

    if divide:

        output = small2large(output)

        output = smooth(output)
    else:
        output = output[0]
    output = (output * 255).astype('uint8')

    print(output.shape)

    cv2.imwrite(os.path.join(output_dir, img_name + '.jpg'), output)

# os.system('mkdir output')

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description='Learning to Paint')
    parser.add_argument('--img_dir', default='./test_images/all/', type=str, help='test image')
    parser.add_argument('--row_divide', default=1, type=int, help='divide the target image to get better resolution')
    parser.add_argument('--col_divide', default=1, type=int, help='divide the target image to get better resolution')
    parser.add_argument('--ckpt', default='./strokepredictor/output/checkpoint-199.pth', type=str)
    parser.add_argument('--output_dir', default='output', type=str)
    args = parser.parse_args()

    os.system('mkdir ' + args.output_dir)

    painter = attn_painter.AttnPainterOilDensity(stroke_num=256).to(device)
    checkpoint_model = torch.load(args.ckpt, map_location='cpu')

    msg = painter.load_state_dict(checkpoint_model['model'], strict=True)
    print(msg)
    total = sum([param.nelement() for param in painter.parameters()])
    print('total:' + str(total))

    all_strokes = []
    file_name = os.listdir(args.img_dir)

    for idx in tqdm(range(0, 499)): #tqdm(range(10000, 10500)):
        torch.cuda.synchronize()
        start = time.time()
        canvas_cnt = args.row_divide * args.col_divide
        T = torch.ones([1, 1, width, width], dtype=torch.float32).to(device)

        img_name = file_name[idx]

        img = cv2.imread(os.path.join(args.img_dir, img_name), cv2.IMREAD_COLOR)
        origin_shape = (img.shape[1], img.shape[0])
        img = cv2.resize(img, (512, 512))

        input_GRAY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gradient_x = cv2.Sobel(input_GRAY, cv2.CV_32FC1, 1, 0, ksize=5)
        gradient_y = cv2.Sobel(input_GRAY, cv2.CV_32FC1, 0, 1, ksize=5)
        gradient_magnitude = np.sqrt(gradient_x ** 2.0 + gradient_y ** 2.0)
        density = cv2.blur(gradient_magnitude, (14, 14))
        masks = cv2.normalize(density, None, 0.01, 1.0, cv2.NORM_MINMAX) # 归一化

        patch_img = cv2.resize(img, (width, width), interpolation= cv2.INTER_LINEAR)
        patch_img = large2small(patch_img) if args.row_divide == 1 and args.col_divide == 1 else large2small_overlap(patch_img, args.row_divide, args.col_divide, overlap=10)
        patch_img = np.transpose(patch_img, (0, 3, 1, 2))
        patch_img = torch.tensor(patch_img).to(device).float() / 255.
        patch_img = TF.resize(patch_img, (224, 224))
        patch_img = TF.normalize(patch_img, mean=channel_mean, std=channel_std)

        patch_mask = cv2.resize(np.repeat(masks[:,:,None], 3,2), (width * args.row_divide, width * args.col_divide), interpolation= cv2.INTER_LINEAR)
        patch_mask = large2small(patch_mask) if args.row_divide == 1 and args.col_divide == 1 else large2small_overlap(patch_mask, args.row_divide, args.col_divide, overlap=10)
        patch_mask = np.transpose(patch_mask, (0, 3, 1, 2))
        patch_mask = torch.tensor(patch_mask).to(device).float()
        patch_mask = TF.resize(patch_mask, (224, 224))

        with torch.no_grad():
            painter.eval()

            preds, gt = painter.real_forward_3(patch_img, patch_mask[:, :1], res=512, col=args.col_divide, row=args.row_divide, overlap=0 if args.row_divide == 1 and args.col_divide == 1 else 10)
            torch.cuda.synchronize()

        save_img(preds, img_name.split('/')[-1], divide=False, output_dir=args.output_dir)


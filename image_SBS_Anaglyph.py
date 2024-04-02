import os
import sys
import cv2
import numpy as np
import subprocess
import platform
from argparse import ArgumentParser
from tqdm import tqdm

import onnxruntime as rt

parser = ArgumentParser()
# input/output files
parser.add_argument("--source", help="path to source image")
parser.add_argument("--output", help="path to result image")

# depth
parser.add_argument('--shift_factor', default=15, type=int, help='Amount of depth')
parser.add_argument('--scale_factor', default=2, type=int, help='Downscale factor for depthmap estimation for faster inference')
#output format side-by-side
parser.add_argument("--sbs", default=True, action="store_true", help="Save output as sbs video")
parser.add_argument("--half_sbs", default=False, action="store_true", help="Save output as half-SBS video")
parser.add_argument("--crosseye", default=False, action="store_true", help="Save output as cross-eye SBS video")

# anaglyph
parser.add_argument("--anaglyph", default=False, action="store_true", help="Save output as red/cyan anaglyph video")
opt = parser.parse_args()

device = 'cuda' # !!fp16 does not work on cpu!! 

# 
from depth.depth import DEPTHMAP
# change path to onnx model here
get_map = DEPTHMAP(model_path="depth/depth_anything_vits14_fp16.onnx", device=device)

# depth
shift_factor = opt.shift_factor

orig_img = cv2.imread(opt.source)
height, width, _ = orig_img.shape

# output format options
if opt.anaglyph:
    opt.sbs = False
    opt.half_sbs = False
    w_out = width

if opt.half_sbs:
    opt.sbs = False
    w_out = width
    
if opt.sbs:
    opt.anaglyph = False
    w_out = width*2

#pass width, height of video frame or any other smaller values for faster inference, (keep multiple of 14 ??)
#w_inf = 518
#h_inf = 518
w_inf = width//opt.scale_factor
h_inf = height//opt.scale_factor

def main():
    orig_img = cv2.imread(opt.source)
    height, width, _ = orig_img.shape
        
    # pass width, height of video frame or any other smaller values for faster inference, keep multiple of 14    
    depth = get_map.process_fp16(orig_img, w_inf, h_inf)
    #cv2.imshow("DepthMap",depth)
    #cv2.waitKey(1)
    
    stereogram = np.zeros_like(orig_img)    
                
    shift = (depth / 255 * shift_factor).astype(int)
    stereogram = np.zeros_like(orig_img)
    shifted_indices = np.maximum(np.arange(orig_img.shape[1])[None, :] - shift[:, :], 0)
    stereogram[np.arange(orig_img.shape[0])[:, None], np.arange(orig_img.shape[1])] = orig_img[np.arange(orig_img.shape[0])[:, None], shifted_indices]
    
    if opt.anaglyph:
        anaglyph = np.zeros_like(orig_img)
        x_left = np.arange(orig_img.shape[1])
        x_right = np.clip(x_left - shift, 0, orig_img.shape[1] - 1)
        anaglyph[:, :, 2] = stereogram[:, :, 2]
        anaglyph[:, :, 0] = orig_img[:, :, 0]
        anaglyph[:, :, 1] = orig_img[:, :, 1]      
        result = anaglyph # red/cyan
    
    if opt.sbs:
        if opt.crosseye:
            result = np.concatenate((orig_img, stereogram), axis=1) #crosseye
        else:
            result = np.concatenate((stereogram, orig_img), axis=1) #parallel
        
    if opt.half_sbs:
        orig_img = cv2.resize(orig_img,(w_out//2, height))
        stereogram = cv2.resize(stereogram,(w_out//2, height))
        if opt.crosseye:
            result = np.concatenate((orig_img, stereogram), axis=1) #crosseye
        else:
            result = np.concatenate((stereogram, orig_img), axis=1) #parallel
    
    cv2.imshow ("Result",result)
    cv2.imwrite(opt.output,result)
    cv2.waitKey()

main()    
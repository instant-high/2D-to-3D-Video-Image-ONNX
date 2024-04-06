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
parser.add_argument("--source", help="path to source video")
parser.add_argument("--result", help="path to result video")
parser.add_argument('--resize', default=1, type=int, help='Resize result video')

# depth
parser.add_argument('--shift_factor', default=10, type=float, help='Amount of depth')
parser.add_argument('--scale_factor', default=2, type=int, help='Downscale factor for depthmap estimation for faster inference')
parser.add_argument("--pop_out", default=False, action="store_true", help="Create 3D video with pop-out")

#output format side-by-side
parser.add_argument("--sbs", default=True, action="store_true", help="Save output as sbs video")
parser.add_argument("--half_sbs", default=False, action="store_true", help="Save output as half-SBS video")
parser.add_argument("--crosseye", default=False, action="store_true", help="Save output as cross-eye SBS video")

# anaglyph
parser.add_argument("--anaglyph", default=False, action="store_true", help="Save output as red/cyan anaglyph video")

# keep audio
parser.add_argument("--audio", default=False, action="store_true", help="Keep audio")
opt = parser.parse_args()

device = 'cuda' # !!fp16 does not work on cpu!! 

# 
from depth.depth import DEPTHMAP
# change path to onnx model here
get_map = DEPTHMAP(model_path="depth/depth_anything_vits14_fp16.onnx", device=device)
#get_map = DEPTHMAP(model_path="depth/depth_anything_vits14.onnx", device=device)

# depth
shift_factor = opt.shift_factor

# video reader
video = cv2.VideoCapture(opt.source)
width_orig = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height_orig = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))    
fps = video.get(cv2.CAP_PROP_FPS)

#resize result (resize before inference)       
resize  = int(opt.resize)/10
width = int(width_orig*resize)
height = int(height_orig*resize)
#print(width,height)
#input("1")
if width %2 !=0 : width = width - 1
if height %2 !=0 : height = height - 1
#print(width,height)
#input("2")

# output format options / (w,h) for cv2.WideoWriter
if opt.anaglyph:
    opt.sbs = False
    opt.half_sbs = False
    w_out = width

if opt.half_sbs:
    opt.sbs = False
    w_out = width
    
if opt.sbs:
    opt.anaglyph = False
    w_out = width *2

# audio option        
if opt.audio:
    writer = cv2.VideoWriter('temp.mp4',cv2.VideoWriter_fourcc('m','p','4','v'), fps, (w_out, height))
else:
    writer = cv2.VideoWriter(opt.result,cv2.VideoWriter_fourcc('m','p','4','v'), fps, (w_out, height))

# pass width, height of video frame or any other values for faster inference
# this has an effect on the shift_factor
# original repo 518,518
# w_inf = 518
# h_inf = 518

#w_inf = width//2
#h_inf = height//2
w_inf = width//opt.scale_factor
h_inf = height//opt.scale_factor

# main inference loop
for frame_idx in tqdm(range(n_frames)):

    ret, orig_img = video.read()
    if not ret:
        break
    
    #resize result (resize before inference)  
    orig_img = cv2.resize(orig_img,(width, height))
    
    # pass width, height of video frame or any other smaller values for faster inference
    # fp16   
    depth = get_map.process_fp16(orig_img, w_inf, h_inf)
    #depth = get_map.process(orig_img, w_inf, h_inf)
    
    #cv2.imshow("DepthMap",depth)
    #cv2.waitKey(1)
    
    # pop-out
    if opt.pop_out:
        stereogram = np.zeros_like(orig_img)    
        shift = (depth / 255 * shift_factor).astype(int)
        stereogram = np.zeros_like(orig_img)
        shifted_indices = np.maximum(np.arange(orig_img.shape[1])[None, :] - shift[:, :], 0)
        stereogram[np.arange(orig_img.shape[0])[:, None], np.arange(orig_img.shape[1])] = orig_img[np.arange(orig_img.shape[0])[:, None], shifted_indices]

    #no pop-out
    else:
        depth_scaled = (depth / np.max(depth)) * 255  # Scale depth values to [0, 255]
        shift = ((255 - depth_scaled) / 255 * shift_factor).astype(int)
        stereogram = np.zeros_like(orig_img)
        shifted_indices = np.maximum(np.arange(orig_img.shape[1])[None, :] + shift[:, :], 0)
        shifted_indices = np.clip(shifted_indices, 0, orig_img.shape[1] - 1)  # Clip indices to avoid out-of-bounds
        stereogram[np.arange(orig_img.shape[0])[:, None], np.arange(orig_img.shape[1])] = orig_img[np.arange(orig_img.shape[0])[:, None], shifted_indices]
        stereogram = np.clip(stereogram, 0, 255)
    
       
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
    
    writer.write(result)
    cv2.imshow ("Result - press ESC to stop",result)
    k = cv2.waitKey(1)
    if k == 27:
        writer.release()
        break

writer.release()

if opt.audio:
    # lossless remuxing audio/video
    # this will crash if source has no audio / has to be fixed
    # result then is temp.mp4
    command = 'ffmpeg.exe -y -vn -i ' + '"' + opt.source + '"' + ' -an -i ' + 'temp.mp4' + ' -c:v copy -acodec libmp3lame -ac 2 -ar 44100 -ab 128000 -map 0:1 -map 1:0 -shortest ' + '"' + opt.result + '"'
    subprocess.call(command, shell=platform.system() != 'Windows')
    os.remove('temp.mp4')
    
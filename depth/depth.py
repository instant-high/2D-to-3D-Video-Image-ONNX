import cv2
import numpy as np
import onnxruntime



class DEPTHMAP:
    def __init__(self, model_path="depth/depth_anything_vits14.onnx", device='cpu'):
        session_options = onnxruntime.SessionOptions()
        session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CPUExecutionProvider"]
        if device == 'cuda':
            providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),"CPUExecutionProvider"]
        self.session = onnxruntime.InferenceSession(model_path, sess_options=session_options, providers=providers)
        self.resolution = self.session.get_inputs()[0].shape[-2:]
        


    def process(self, img, w, h):
        # preprocess
        height, width = img.shape[:2]
        img = cv2.resize(img, (w, h)) # dynamic / original = 518x518 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        img = img.astype(np.float32)  
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, axis=0).astype(np.float32)

        # Run inference to obtain depth_map
        depth_map = self.session.run(None, {"image": img})[0]

        # postprocess
        depth_map = cv2.resize(depth_map[0, 0], (width, height))
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255.0
        depth_map = depth_map.astype(np.uint8)    
        
        return depth_map

    
    def process_fp16(self, img, w, h):
        # preprocess
        height, width = img.shape[:2]
        img = cv2.resize(img, (w, h)) # dynamic / original = 518x518 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        img = img.astype(np.float16)  
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, axis=0).astype(np.float16)

        # Run inference to obtain depth_map
        depth_map = self.session.run(None, {"image": img})[0]

        # postprocess
        depth_map = depth_map.astype(np.float32)
        depth_map = cv2.resize(depth_map[0, 0], (width, height))
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255.0
        depth_map = depth_map.astype(np.uint8)    
        
        return depth_map        
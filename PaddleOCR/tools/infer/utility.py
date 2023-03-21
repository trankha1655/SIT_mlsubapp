# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import sys
import platform
import cv2
import numpy as np
import paddle
from PIL import Image, ImageDraw, ImageFont
import math
from paddle import inference
import time
import random
from ppocr.utils.logging import get_logger


def str2bool(v):
    return v.lower() in ("true", "t", "1")


def init_args():
    parser = argparse.ArgumentParser()
    # params for prediction engine
    parser.add_argument("--use_gpu", type=str2bool, default=False)
    parser.add_argument("--use_xpu", type=str2bool, default=False)
    parser.add_argument("--use_npu", type=str2bool, default=False)
    parser.add_argument("--ir_optim", type=str2bool, default=True)
    parser.add_argument("--use_tensorrt", type=str2bool, default=False)
    parser.add_argument("--min_subgraph_size", type=int, default=15)
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--gpu_mem", type=int, default=500)
    parser.add_argument("--gpu_id", type=int, default=0)

    # params for text detector
    parser.add_argument("--image_dir", type=str, default='../../MBA/origin/')
    parser.add_argument("--page_num", type=int, default=0)
    parser.add_argument("--det_algorithm", type=str, default='DB')
    parser.add_argument("--det_model_dir", type=str, default='../../weight/Det_DB/Student/inference.pdmodel')
    parser.add_argument("--det_limit_side_len", type=float, default=960)
    parser.add_argument("--det_limit_type", type=str, default='max')
    parser.add_argument("--det_box_type", type=str, default='quad')

    # DB parmas
    parser.add_argument("--det_db_thresh", type=float, default=0.3)
    parser.add_argument("--det_db_box_thresh", type=float, default=0.6)
    parser.add_argument("--det_db_unclip_ratio", type=float, default=1.5)
    parser.add_argument("--max_batch_size", type=int, default=10)
    parser.add_argument("--use_dilation", type=str2bool, default=False)
    parser.add_argument("--det_db_score_mode", type=str, default="fast")

    # EAST parmas
    parser.add_argument("--det_east_score_thresh", type=float, default=0.8)
    parser.add_argument("--det_east_cover_thresh", type=float, default=0.1)
    parser.add_argument("--det_east_nms_thresh", type=float, default=0.2)

    # SAST parmas
    parser.add_argument("--det_sast_score_thresh", type=float, default=0.5)
    parser.add_argument("--det_sast_nms_thresh", type=float, default=0.2)

    # PSE parmas
    parser.add_argument("--det_pse_thresh", type=float, default=0)
    parser.add_argument("--det_pse_box_thresh", type=float, default=0.85)
    parser.add_argument("--det_pse_min_area", type=float, default=16)
    parser.add_argument("--det_pse_scale", type=int, default=1)

    # FCE parmas
    parser.add_argument("--scales", type=list, default=[8, 16, 32])
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--fourier_degree", type=int, default=5)

    # params for text recognizer
    parser.add_argument("--rec_algorithm", type=str, default='SVTR_LCNet')
    parser.add_argument("--rec_model_dir", type=str, default='../../weight/Rec_SVTR/PPOCRv3_en/svtr_tiny2.onnx')
    parser.add_argument("--rec_image_inverse", type=str2bool, default=True)
    parser.add_argument("--rec_image_shape", type=str, default="3, 96, 640")
    parser.add_argument("--rec_batch_num", type=int, default=6)
    parser.add_argument("--max_text_length", type=int, default=25)
    parser.add_argument(
        "--rec_char_dict_path",
        type=str,
        default="./ppocr/utils/en_dict.txt")
    parser.add_argument("--use_space_char", type=str2bool, default=True)
    parser.add_argument(
        "--vis_font_path", type=str, default="./doc/fonts/simfang.ttf")
    parser.add_argument("--drop_score", type=float, default=0.5)

    # params for e2e
    parser.add_argument("--e2e_algorithm", type=str, default='PGNet')
    parser.add_argument("--e2e_model_dir", type=str)
    parser.add_argument("--e2e_limit_side_len", type=float, default=768)
    parser.add_argument("--e2e_limit_type", type=str, default='max')

    # PGNet parmas
    parser.add_argument("--e2e_pgnet_score_thresh", type=float, default=0.5)
    parser.add_argument(
        "--e2e_char_dict_path", type=str, default="./ppocr/utils/ic15_dict.txt")
    parser.add_argument("--e2e_pgnet_valid_set", type=str, default='totaltext')
    parser.add_argument("--e2e_pgnet_mode", type=str, default='fast')

    # params for text classifier
    parser.add_argument("--use_angle_cls", type=str2bool, default=False)
    parser.add_argument("--cls_model_dir", type=str)
    parser.add_argument("--cls_image_shape", type=str, default="3, 48, 192")
    parser.add_argument("--label_list", type=list, default=['0', '180'])
    parser.add_argument("--cls_batch_num", type=int, default=6)
    parser.add_argument("--cls_thresh", type=float, default=0.9)

    parser.add_argument("--enable_mkldnn", type=str2bool, default=False)
    parser.add_argument("--cpu_threads", type=int, default=10)
    parser.add_argument("--use_pdserving", type=str2bool, default=False)
    parser.add_argument("--warmup", type=str2bool, default=False)

    # SR parmas
    parser.add_argument("--sr_model_dir", type=str)
    parser.add_argument("--sr_image_shape", type=str, default="3, 32, 128")
    parser.add_argument("--sr_batch_num", type=int, default=1)

    #
    parser.add_argument(
        "--draw_img_save_dir", type=str, default="./inference_results/")
    parser.add_argument("--save_crop_res", type=str2bool, default=False)
    parser.add_argument("--crop_res_save_dir", type=str, default="./output")

    # multi-process
    parser.add_argument("--use_mp", type=str2bool, default=False)
    parser.add_argument("--total_process_num", type=int, default=1)
    parser.add_argument("--process_id", type=int, default=0)

    parser.add_argument("--benchmark", type=str2bool, default=False)
    parser.add_argument("--save_log_path", type=str, default="./log_output/")

    parser.add_argument("--show_log", type=str2bool, default=True)
    parser.add_argument("--use_onnx", type=str2bool, default=False)
    parser.add_argument("--use_openvino", type=str2bool, default=True)
    return parser


def parse_args(Mmain=False):
    parser = init_args()

    if not Mmain:
        inference_args_dict = {}
        for action in parser._actions:
            inference_args_dict[action.dest] = action.default
        return argparse.Namespace(**inference_args_dict)
    
    
    return parser.parse_args()



def create_predictor(args, mode, logger):
    if mode == "det":
        model_dir = args.det_model_dir
    elif mode == 'cls':
        model_dir = args.cls_model_dir
    elif mode == 'rec':
        model_dir = args.rec_model_dir
    elif mode == 'table':
        model_dir = args.table_model_dir
    elif mode == 'ser':
        model_dir = args.ser_model_dir
    elif mode == 're':
        model_dir = args.re_model_dir
    elif mode == "sr":
        model_dir = args.sr_model_dir
    elif mode == 'layout':
        model_dir = args.layout_model_dir
    else:
        model_dir = args.e2e_model_dir

    if model_dir is None:
        logger.info("not find {} model file path {}".format(mode, model_dir))
        sys.exit(0)
    if args.use_onnx:
        import onnxruntime as ort
        model_file_path = model_dir
        if not os.path.exists(model_file_path):
            raise ValueError("not find model file path {}".format(
                model_file_path))
        sess = ort.InferenceSession(model_file_path)
        #print(sess, sess.get_inputs()[0])
        return sess, sess.get_inputs()[0], None, None
    
    elif args.use_openvino:
        try:
            from openvino.runtime import Core
        except:
            raise ValueError('Please import higer openvino version.')
        ie = Core()
        

        
        model_file_path = model_dir
        if not os.path.exists(model_file_path):
            raise ValueError("not find model file path {}".format(
                model_file_path))
        
        model_onnx = ie.read_model(model=model_file_path)
        compiled_model = ie.compile_model(model=model_onnx, device_name="CPU")
        #request = compiled_model.create_infer_request()
        input, output = compiled_model.inputs[0], compiled_model.outputs[0]
        # print(input.any_name)
        # tt = compiled_model.get_output_tensor()
        # print(tt)
        # print(k)

        return compiled_model.create_infer_request(), input, output, None
    

    else:
        file_names = ['model', 'inference']
        for file_name in file_names:
            model_file_path = '{}/{}.pdmodel'.format(model_dir, file_name)
            params_file_path = '{}/{}.pdiparams'.format(model_dir, file_name)
            if os.path.exists(model_file_path) and os.path.exists(
                    params_file_path):
                break
        if not os.path.exists(model_file_path):
            raise ValueError(
                "not find model.pdmodel or inference.pdmodel in {}".format(
                    model_dir))
        if not os.path.exists(params_file_path):
            raise ValueError(
                "not find model.pdiparams or inference.pdiparams in {}".format(
                    model_dir))

        config = inference.Config(model_file_path, params_file_path)

        if hasattr(args, 'precision'):
            if args.precision == "fp16" and args.use_tensorrt:
                precision = inference.PrecisionType.Half
            elif args.precision == "int8":
                precision = inference.PrecisionType.Int8
            else:
                precision = inference.PrecisionType.Float32
        else:
            precision = inference.PrecisionType.Float32

        if args.use_gpu:
            gpu_id = get_infer_gpuid()
            if gpu_id is None:
                logger.warning(
                    "GPU is not found in current device by nvidia-smi. Please check your device or ignore it if run on jetson."
                )
            config.enable_use_gpu(args.gpu_mem, args.gpu_id)
            if args.use_tensorrt:
                config.enable_tensorrt_engine(
                    workspace_size=1 << 30,
                    precision_mode=precision,
                    max_batch_size=args.max_batch_size,
                    min_subgraph_size=args.
                    min_subgraph_size,  # skip the minmum trt subgraph
                    use_calib_mode=False)

                # collect shape
                trt_shape_f = os.path.join(model_dir,
                                           f"{mode}_trt_dynamic_shape.txt")

                if not os.path.exists(trt_shape_f):
                    config.collect_shape_range_info(trt_shape_f)
                    logger.info(
                        f"collect dynamic shape info into : {trt_shape_f}")
                try:
                    config.enable_tuned_tensorrt_dynamic_shape(trt_shape_f,
                                                               True)
                except Exception as E:
                    logger.info(E)
                    logger.info("Please keep your paddlepaddle-gpu >= 2.3.0!")

        elif args.use_npu:
            config.enable_npu()
        elif args.use_xpu:
            config.enable_xpu(10 * 1024 * 1024)
        else:
            config.disable_gpu()
            if args.enable_mkldnn:
                # cache 10 different shapes for mkldnn to avoid memory leak
                config.set_mkldnn_cache_capacity(10)
                config.enable_mkldnn()
                if args.precision == "fp16":
                    config.enable_mkldnn_bfloat16()
                if hasattr(args, "cpu_threads"):
                    config.set_cpu_math_library_num_threads(args.cpu_threads)
                else:
                    # default cpu threads as 10
                    config.set_cpu_math_library_num_threads(10)
        # enable memory optim
        config.enable_memory_optim()
        config.disable_glog_info()
        config.delete_pass("conv_transpose_eltwiseadd_bn_fuse_pass")
        config.delete_pass("matmul_transpose_reshape_fuse_pass")
        if mode == 're':
            config.delete_pass("simplify_with_basic_ops_pass")
        if mode == 'table':
            config.delete_pass("fc_fuse_pass")  # not supported for table
        config.switch_use_feed_fetch_ops(False)
        config.switch_ir_optim(True)

        # create predictor
        predictor = inference.create_predictor(config)
        input_names = predictor.get_input_names()
        if mode in ['ser', 're']:
            input_tensor = []
            for name in input_names:
                input_tensor.append(predictor.get_input_handle(name))
        else:
            for name in input_names:
                input_tensor = predictor.get_input_handle(name)
        output_tensors = get_output_tensors(args, mode, predictor)
        print(config)
        return predictor, input_tensor, output_tensors, config


def get_output_tensors(args, mode, predictor):
    output_names = predictor.get_output_names()
    output_tensors = []
    if mode == "rec" and args.rec_algorithm in ["CRNN", "SVTR_LCNet"]:
        output_name = 'softmax_0.tmp_0'
        if output_name in output_names:
            return [predictor.get_output_handle(output_name)]
        else:
            for output_name in output_names:
                output_tensor = predictor.get_output_handle(output_name)
                output_tensors.append(output_tensor)
    else:
        for output_name in output_names:
            output_tensor = predictor.get_output_handle(output_name)
            output_tensors.append(output_tensor)
    return output_tensors


def get_infer_gpuid():
    sysstr = platform.system()
    if sysstr == "Windows":
        return 0

    if not paddle.fluid.core.is_compiled_with_rocm():
        cmd = "env | grep CUDA_VISIBLE_DEVICES"
    else:
        cmd = "env | grep HIP_VISIBLE_DEVICES"
    env_cuda = os.popen(cmd).readlines()
    if len(env_cuda) == 0:
        return 0
    else:
        gpu_id = env_cuda[0].strip().split("=")[1]
        return int(gpu_id[0])


class My_PreProcess:
    def __init__(self):
        self.lower = np.array([0, 53, 189])
        self.upper = np.array([179, 255, 255])
    def rotate_one_side(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        result = cv2.bitwise_and(image, image, mask=mask)
        
        gray = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

        H, W = image.shape[:2]
        half_W = W//2
        half_left = gray[:, :half_W].sum()
        half_right = gray[:, half_W: ].sum()
        if half_right > half_left:
            return cv2.rotate(image, cv2.ROTATE_180)
        
        return image

    def crop_mba_area(self, origin):
        img = self.read_image(origin)
        tl, tr, bl, br = self.check_type(origin)
        #print(tl, tr, bl, br)
        #print("Find 4 points: %.4f" % (time.time()-st), end= ' ')
        if (tl is None or tr is None or bl is None or br is None):
            #print(file)
            pass
            return None 
        #note: in 0.2 scale
        tl_new, new_H, new_W = self.get_HW_box(tl, tr, bl, br)
        # note: in normal scale
        tl_crop, br_crop, color_ = self.get_point_crop(tl_new, new_H, new_W, img2cir=img)

        #print(tl_crop, br_crop, color_)

        img_crop = origin[tl_crop[0]: br_crop[0], tl_crop[1]: br_crop[1]]       

        img = cv2.rotate(img_crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img

    def find_center_contour(self, img, H=0, W=0):
        #st = time.time()
        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        ans = None
        for ct in contours:
            if cv2.contourArea(ct) > max_area:
                max_area = cv2.contourArea(ct)
                ans = ct
        if ans is None:
            return None
        M = cv2.moments(ans)
        #print('Find 1 contour: %.4f' % (time.time()-st), end=' ')
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            
            return cx + W, cy + H
    

    def check_type(self, image):
        lower = np.array([0, 30, 80])
        upper = np.array([18, 255, 255])

        H,W = image.shape[:2]
        H = int(0.2*H)
        W = int(0.2*W)
        img = cv2.resize(image, (W,H) )

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(img, img, mask=mask)


        gray = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.blur(gray, (3, 3))

        gray_blurred = np.where(gray_blurred > 42, gray_blurred, 0)
        mask2 = np.ones_like(gray_blurred)
        mask2[88:513, 100:709] = 0
        gray_blurred = gray_blurred*mask2
        if gray_blurred.sum() < 180000:
            return False

        tl = self.find_center_contour(gray_blurred[0:80, 0:152])
        tr = self.find_center_contour(gray_blurred[0:80, 690:W], W=690)
        bl = self.find_center_contour(gray_blurred[500:H, 0:176], H=500)
        br = self.find_center_contour(gray_blurred[500:H, 690:W], H=500, W=690)
        
        return tl, tr, bl, br


    def get_HW_box(self, tl, tr, bl, br):
        """
            This function to get anchor point (top left) and H, W from 4 red points on image.
            input: 4 points (top left, top right, bottom left, bottom right) folow (y,x) cv2 format
            output: top_left point, H, W
        """
        tl_x = tl[1] + tr[1] 
        tl_y = tl[0] + bl[0]

        br_x = br[1] + bl[1]
        br_y = br[0] + tr[0]
                #top left point, H, W
        return (int(tl_x/2), int(tl_y/2)), int((br_x-tl_x)/2), int((br_y-tl_y)/2)


    def find_circle(self, image, ratio=0.5):
        #image is normal scale 
        #we need to downscale to apply algorithm
        H,W = image.shape[:2]
        H= int(H*ratio)
        W= int(W*ratio)
        img = cv2.resize(image, (W, H))

        # Convert to HSV format and color threshold
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([50, 255, 255]))
        result = cv2.bitwise_and(img, img, mask=mask)

        gray = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.blur(gray, (13, 13))

        gray_blurred = np.where(gray_blurred > 63, gray_blurred, 0)

        detected_circles = cv2.HoughCircles(gray_blurred, 
                        cv2.HOUGH_GRADIENT, 1, 20, param1 = 50,
                    param2 = 30, minRadius = 53, maxRadius = 79)
        point = []
        #find circle
        if detected_circles is not None:

            # Convert the circle parameters a, b and r to integers.
            detected_circles = np.uint16(np.around(detected_circles))
            #print('nums', len(detected_circles[0, :]))
            for pt in detected_circles[0, :]:
                a, b, r = pt[0], pt[1], pt[2]
        
                if abs(a-W/2) > 170:
                    print('next', a-W/2)
                    continue
                cv2.circle(img, (a,b), r, [255, 255, 0], 3, -1)
                
                point.append([a, b])
            
        #convert to normal scale
        cir_point = np.mean(point, axis=0)/ratio
        #print(cir_point)

        return cir_point.astype(np.int32)

    def read_image(self, path):
        if isinstance(path, str):
            image = cv2.imread(path)
        else:
            image = path
        H,W = image.shape[:2]
        H = int(0.2*H)
        W = int(0.2*W)
        img = cv2.resize(image, (W,H) )

        if isinstance(path, str):
            return img, image
        else:
            return img
        

    def get_point_crop(self, tl, H, W, return_ratio=0.2, img2cir= None, cir_ratio=0.5):
        """
            This function to generate 2 new point to crop ROI.
            2 new point depend on my Ratio by observation.
            input: top_left point, H, W
        """
        #   ratio * new_size + anchor_point  = new_point
        tl_x = (160/580 * H + tl[0])/return_ratio
        tl_y = (400/730 * W + tl[1])/return_ratio

        size_x = 260/580 * H /return_ratio
        size_y = 230/580 * H /return_ratio

        br_x = tl_x + size_x
        br_y = tl_y + size_y 

        tl_x, tl_y, br_x, br_y = int(tl_x), int(tl_y), int(br_x), int(br_y)

        if img2cir is not None:
            #up scale to normal
            h, w = img2cir.shape[:2]
            h = int(h/return_ratio)
            w = int(w/return_ratio)
            img = cv2.resize(img2cir, (w,h))
            #find circle, return coor in normoal scale
            st = time.time()
            cir_point = self.find_circle(img[tl_x: br_x, tl_y: br_y], ratio=cir_ratio)
            #print("Find circle: %.4f" % (time.time()-st), end=' ')
            try:
                
                cir_x = cir_point[1] + tl_x
                cir_y = cir_point[0] + tl_y
            
                cv2.circle(img2cir, (cir_y//5, cir_x//5), 4, [0, 255, 255], 5, -1)
                #cv2.imshow('cir', img2cir)
                return (int(cir_x- size_x/2), int(cir_y- size_y/3.5)), ( int(cir_x+ size_x/2), int(cir_y + size_y/1.8)), [0,0,255]
            except:
                pass
            

        return (int(tl_x), int(tl_y)), (int(br_x), int(br_y)), [255,0,0]

    def __call__(self, path):
        if isinstance(path, str):
            image = cv2.imread(path)
        else:
            image = path
        image = self.rotate_one_side(image)
        crop_image = self.crop_mba_area(image)
        return crop_image


def my_imread(file):
    img = cv2.imread(file)
    if 'Direct_1' in file: 
        img = img[810:810+1210, 850:850+1050]
        print('[CROP] 1', img.shape)
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    
    if 'Direct_2' in file:
        img = img[950: 950+1100, 2175:2175+1150]
        print('[CROP] 2')
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img

def draw_e2e_res(dt_boxes, strs, img_path):
    src_im = my_imread(img_path)
    for box, str in zip(dt_boxes, strs):
        box = box.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
        cv2.putText(
            src_im,
            str,
            org=(int(box[0, 0, 0]), int(box[0, 0, 1])),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=0.7,
            color=(0, 255, 0),
            thickness=1)
    return src_im


def draw_text_det_res(dt_boxes, img):
    for box in dt_boxes:
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        cv2.polylines(img, [box], True, color=(255, 255, 0), thickness=2)
    return img


def resize_img(img, input_size=600):
    """
    resize img and limit the longest side of the image to input_size
    """
    img = np.array(img)
    im_shape = img.shape
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(input_size) / float(im_size_max)
    img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale)
    return img


def draw_ocr(image,
             boxes,
             txts=None,
             scores=None,
             drop_score=0.5,
             font_path="./doc/fonts/simfang.ttf"):
    """
    Visualize the results of OCR detection and recognition
    args:
        image(Image|array): RGB image
        boxes(list): boxes with shape(N, 4, 2)
        txts(list): the texts
        scores(list): txxs corresponding scores
        drop_score(float): only scores greater than drop_threshold will be visualized
        font_path: the path of font which is used to draw text
    return(array):
        the visualized img
    """
    if scores is None:
        scores = [1] * len(boxes)
    box_num = len(boxes)
    for i in range(box_num):
        if scores is not None and (scores[i] < drop_score or
                                   math.isnan(scores[i])):
            continue
        box = np.reshape(np.array(boxes[i]), [-1, 1, 2]).astype(np.int64)
        image = cv2.polylines(np.array(image), [box], True, (255, 0, 0), 2)
    if txts is not None:
        img = np.array(resize_img(image, input_size=600))
        txt_img = text_visual(
            txts,
            scores,
            img_h=img.shape[0],
            img_w=600,
            threshold=drop_score,
            font_path=font_path)
        img = np.concatenate([np.array(img), np.array(txt_img)], axis=1)
        return img
    return image


def draw_ocr_box_txt(image,
                     boxes,
                     txts=None,
                     scores=None,
                     drop_score=0.5,
                     font_path="./doc/fonts/simfang.ttf"):
    h, w = image.height, image.width
    img_left = image.copy()
    img_right = np.ones((h, w, 3), dtype=np.uint8) * 255
    random.seed(0)

    draw_left = ImageDraw.Draw(img_left)
    if txts is None or len(txts) != len(boxes):
        txts = [None] * len(boxes)
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        if scores is not None and scores[idx] < drop_score:
            continue
        color = (random.randint(0, 255), random.randint(0, 255),
                 random.randint(0, 255))
        draw_left.polygon(box, fill=color)
        img_right_text = draw_box_txt_fine((w, h), box, txt, font_path)
        pts = np.array(box, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_right_text, [pts], True, color, 1)
        img_right = cv2.bitwise_and(img_right, img_right_text)
    img_left = Image.blend(image, img_left, 0.2)
    img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(Image.fromarray(img_right), (w, 0, w * 2, h))
    return np.array(img_show)


def draw_box_txt_fine(img_size, box, txt, font_path="./doc/fonts/simfang.ttf"):
    box_height = int(
        math.sqrt((box[0][0] - box[3][0])**2 + (box[0][1] - box[3][1])**2))
    box_width = int(
        math.sqrt((box[0][0] - box[1][0])**2 + (box[0][1] - box[1][1])**2))

    if box_height > 2 * box_width and box_height > 30:
        img_text = Image.new('RGB', (box_height, box_width), (255, 255, 255))
        draw_text = ImageDraw.Draw(img_text)
        if txt:
            font = create_font(txt, (box_height, box_width), font_path)
            draw_text.text([0, 0], txt, fill=(0, 0, 0), font=font)
        img_text = img_text.transpose(Image.ROTATE_270)
    else:
        img_text = Image.new('RGB', (box_width, box_height), (255, 255, 255))
        draw_text = ImageDraw.Draw(img_text)
        if txt:
            font = create_font(txt, (box_width, box_height), font_path)
            draw_text.text([0, 0], txt, fill=(0, 0, 0), font=font)

    pts1 = np.float32(
        [[0, 0], [box_width, 0], [box_width, box_height], [0, box_height]])
    pts2 = np.array(box, dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts1, pts2)

    img_text = np.array(img_text, dtype=np.uint8)
    img_right_text = cv2.warpPerspective(
        img_text,
        M,
        img_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255))
    return img_right_text


def create_font(txt, sz, font_path="./doc/fonts/simfang.ttf"):
    font_size = int(sz[1] * 0.99)
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    length = font.getsize(txt)[0]
    if length > sz[0]:
        font_size = int(font_size * sz[0] / length)
        font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    return font


def str_count(s):
    """
    Count the number of Chinese characters,
    a single English character and a single number
    equal to half the length of Chinese characters.
    args:
        s(string): the input of string
    return(int):
        the number of Chinese characters
    """
    import string
    count_zh = count_pu = 0
    s_len = len(s)
    en_dg_count = 0
    for c in s:
        if c in string.ascii_letters or c.isdigit() or c.isspace():
            en_dg_count += 1
        elif c.isalpha():
            count_zh += 1
        else:
            count_pu += 1
    return s_len - math.ceil(en_dg_count / 2)


def text_visual(texts,
                scores,
                img_h=400,
                img_w=600,
                threshold=0.,
                font_path="./doc/simfang.ttf"):
    """
    create new blank img and draw txt on it
    args:
        texts(list): the text will be draw
        scores(list|None): corresponding score of each txt
        img_h(int): the height of blank img
        img_w(int): the width of blank img
        font_path: the path of font which is used to draw text
    return(array):
    """
    if scores is not None:
        assert len(texts) == len(
            scores), "The number of txts and corresponding scores must match"

    def create_blank_img():
        blank_img = np.ones(shape=[img_h, img_w], dtype=np.int8) * 255
        blank_img[:, img_w - 1:] = 0
        blank_img = Image.fromarray(blank_img).convert("RGB")
        draw_txt = ImageDraw.Draw(blank_img)
        return blank_img, draw_txt

    blank_img, draw_txt = create_blank_img()

    font_size = 20
    txt_color = (0, 0, 0)
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")

    gap = font_size + 5
    txt_img_list = []
    count, index = 1, 0
    for idx, txt in enumerate(texts):
        index += 1
        if scores[idx] < threshold or math.isnan(scores[idx]):
            index -= 1
            continue
        first_line = True
        while str_count(txt) >= img_w // font_size - 4:
            tmp = txt
            txt = tmp[:img_w // font_size - 4]
            if first_line:
                new_txt = str(index) + ': ' + txt
                first_line = False
            else:
                new_txt = '    ' + txt
            draw_txt.text((0, gap * count), new_txt, txt_color, font=font)
            txt = tmp[img_w // font_size - 4:]
            if count >= img_h // gap - 1:
                txt_img_list.append(np.array(blank_img))
                blank_img, draw_txt = create_blank_img()
                count = 0
            count += 1
        if first_line:
            new_txt = str(index) + ': ' + txt + '   ' + '%.3f' % (scores[idx])
        else:
            new_txt = "  " + txt + "  " + '%.3f' % (scores[idx])
        draw_txt.text((0, gap * count), new_txt, txt_color, font=font)
        # whether add new blank img or not
        if count >= img_h // gap - 1 and idx + 1 < len(texts):
            txt_img_list.append(np.array(blank_img))
            blank_img, draw_txt = create_blank_img()
            count = 0
        count += 1
    txt_img_list.append(np.array(blank_img))
    if len(txt_img_list) == 1:
        blank_img = np.array(txt_img_list[0])
    else:
        blank_img = np.concatenate(txt_img_list, axis=1)
    return np.array(blank_img)


def base64_to_cv2(b64str):
    import base64
    data = base64.b64decode(b64str.encode('utf8'))
    data = np.frombuffer(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data


def draw_boxes(image, boxes, scores=None, drop_score=0.5):
    if scores is None:
        scores = [1] * len(boxes)
    for (box, score) in zip(boxes, scores):
        if score < drop_score:
            continue
        box = np.reshape(np.array(box), [-1, 1, 2]).astype(np.int64)
        image = cv2.polylines(np.array(image), [box], True, (255, 0, 0), 2)
    return image


def get_rotate_crop_image(img, points):
    '''
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    '''
    assert len(points) == 4, "shape of points must be 4*2"
    points[0, 0] -= 10  
    points[3, 0] -= 10  
    points[1, 0] += 10  
    points[2, 0] += 10  

    points[0, 1] -= 10  
    points[1, 1] -= 10  
    points[3, 1] += 10  
    points[2, 1] += 10 

    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


def get_minarea_rect_crop(img, points):
    bounding_box = cv2.minAreaRect(np.array(points).astype(np.int32))
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_a, index_b, index_c, index_d = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_a = 0
        index_d = 1
    else:
        index_a = 1
        index_d = 0
    if points[3][1] > points[2][1]:
        index_b = 2
        index_c = 3
    else:
        index_b = 3
        index_c = 2

    box = [points[index_a], points[index_b], points[index_c], points[index_d]]
    crop_img = get_rotate_crop_image(img, np.array(box))
    return crop_img


def check_gpu(use_gpu):
    if use_gpu and not paddle.is_compiled_with_cuda():
        use_gpu = False
    return use_gpu


if __name__ == '__main__':
    pass

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
import os
import sys
import importlib



__dir__ = os.path.dirname(__file__)

import paddle


sys.path.append(os.path.join(__dir__, ''))
os.environ['PATH'] =   __dir__
print(__dir__)
os.chdir(__dir__)


#os.environ["FLAGS_allocator_strategy"] = 'auto_growth'





# tools = importlib.import_module('.', 'tools')
# ppocr = importlib.import_module('.', 'ppocr')


__all__ = ['OCR']
VERSION = '2.6.1.0'
BASE_DIR = os.path.expanduser("~/.ocr/")

import cv2
import copy
import numpy as np
import json
import time
import logging
from PIL import Image

import tools.infer.utility as utility
import tools.infer.predict_rec as predict_rec
import tools.infer.predict_det as predict_det
import tools.infer.predict_cls as predict_cls
from ppocr.utils.utility import get_image_file_list, check_and_read
from ppocr.utils.logging import get_logger
from tools.infer.utility import draw_ocr_box_txt, get_rotate_crop_image, get_minarea_rect_crop
logger = get_logger()





class TextSystem(object):
    def __init__(self, args):
        if not args.show_log:
            logger.setLevel(logging.INFO)

        self.text_detector = predict_det.TextDetector(args, utility.My_PreProcess())
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.use_angle_cls = args.use_angle_cls
        self.drop_score = args.drop_score
        if self.use_angle_cls:
            self.text_classifier = predict_cls.TextClassifier(args)

        self.args = args
        self.crop_image_res_index = 0
        self.clahe2 = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4,4))
        

    def draw_crop_rec_res(self, output_dir, img_crop_list, rec_res):
        os.makedirs(output_dir, exist_ok=True)
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite(
                os.path.join(output_dir,
                             f"mg_crop_{bno+self.crop_image_res_index}.jpg"),
                img_crop_list[bno])
            logger.debug(f"{bno}, {rec_res[bno]}")
        self.crop_image_res_index += bbox_num

    def __call__(self, img, cls=True):
        time_dict = {'det': 0, 'rec': 0, 'csl': 0, 'all': 0}
        start = time.time()
        ori_im = img.copy()
        dt_boxes, elapse = self.text_detector(img)
        time_dict['det'] = elapse
        logger.debug("dt_boxes num : {}, elapse : {}".format(
            len(dt_boxes), elapse))
        if dt_boxes is None:
            return None, None
        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            if self.args.det_box_type == "quad":
                img_crop = get_rotate_crop_image(ori_im, tmp_box)
            else:
                img_crop = get_minarea_rect_crop(ori_im, tmp_box)
            if False:
                img_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
                img_crop = self.clahe2.apply(img_crop)
                img_crop = cv2.cvtColor(img_crop, cv2.COLOR_GRAY2RGB)
            img_crop_list.append(img_crop)
        if self.use_angle_cls and cls:
            img_crop_list, angle_list, elapse = self.text_classifier(
                img_crop_list)
            time_dict['cls'] = elapse
            logger.debug("cls num  : {}, elapse : {}".format(
                len(img_crop_list), elapse))

        rec_res, elapse = self.text_recognizer(img_crop_list)
        time_dict['rec'] = elapse
        logger.debug("rec_res num  : {}, elapse : {}".format(
            len(rec_res), elapse))
        if self.args.save_crop_res:
            self.draw_crop_rec_res(self.args.crop_res_save_dir, img_crop_list,
                                   rec_res)
        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)
        end = time.time()
        time_dict['all'] = end - start
        return filter_boxes, filter_rec_res, time_dict


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                    (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes


def main(args):
    image_file_list, _ = get_image_file_list(args.image_dir)
    image_file_list = image_file_list[args.process_id::args.total_process_num]
    text_sys = TextSystem(args)
    is_visualize = True
    font_path = args.vis_font_path
    drop_score = args.drop_score
    draw_img_save_dir = args.draw_img_save_dir
    os.makedirs(draw_img_save_dir, exist_ok=True)
    save_results = []

    logger.info(
        "In PP-OCRv3, rec_image_shape parameter defaults to '3, 48, 320', "
        "if you are using recognition model with PP-OCRv2 or an older version, please set --rec_image_shape='3,32,320"
    )

    preprocess = utility.My_PreProcess()
    # warm up 10 times
    if args.warmup:
        img = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
        for i in range(10):
            res = text_sys(img)

    total_time = 0
    cpu_mem, gpu_mem, gpu_util = 0, 0, 0
    _st = time.time()
    count = 0
    #print(image_file_list)
    for idx, image_file in enumerate(image_file_list):
        #print(image_file)
        img, flag_gif, flag_pdf = check_and_read(image_file)
        if not flag_gif and not flag_pdf:
            img = preprocess(image_file)
        if not flag_pdf:
            if img is None:
                logger.debug("error in loading image:{}".format(image_file))
                continue
            imgs = [img]
        else:
            page_num = args.page_num
            if page_num > len(img) or page_num == 0:
                page_num = len(img)
            imgs = img[:page_num]
        for index, img in enumerate(imgs):
            starttime = time.time()
            dt_boxes, rec_res, time_dict = text_sys(img)
            elapse = time.time() - starttime
            total_time += elapse
            if len(imgs) > 1:
                logger.debug(
                    str(idx) + '_' + str(index) + "  Predict time of %s: %.3fs"
                    % (image_file, elapse))
            else:
                logger.debug(
                    str(idx) + "  Predict time of %s: %.3fs" % (image_file,
                                                                elapse))
            for text, score in rec_res:
                logger.debug("{}, {:.3f}".format(text, score))

            res = [{
                "transcription": rec_res[i][0],
                "points": np.array(dt_boxes[i]).astype(np.int32).tolist(),
            } for i in range(len(dt_boxes))]
            if len(imgs) > 1:
                save_pred = os.path.basename(image_file) + '_' + str(
                    index) + "\t" + json.dumps(
                        res, ensure_ascii=False) + "\n"
            else:
                save_pred = os.path.basename(image_file) + "\t" + json.dumps(
                    res, ensure_ascii=False) + "\n"
            save_results.append(save_pred)

            if is_visualize:
                image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                boxes = dt_boxes
                txts = [rec_res[i][0] for i in range(len(rec_res))]
                scores = [rec_res[i][1] for i in range(len(rec_res))]

                draw_img = draw_ocr_box_txt(
                    image,
                    boxes,
                    txts,
                    scores,
                    drop_score=drop_score,
                    font_path=font_path)
                if flag_gif:
                    save_file = image_file[:-3] + "png"
                elif flag_pdf:
                    save_file = image_file.replace('.pdf',
                                                   '_' + str(index) + '.png')
                else:
                    save_file = image_file
                cv2.imwrite(
                    os.path.join(draw_img_save_dir,
                                 os.path.basename(save_file)),
                    draw_img[:, :, ::-1])
                logger.debug("The visualized image saved in {}".format(
                    os.path.join(draw_img_save_dir, os.path.basename(
                        save_file))))

    logger.info("The predict total time is {}".format(time.time() - _st))
    if args.benchmark:
        text_sys.text_detector.autolog.report()
        text_sys.text_recognizer.autolog.report()

    with open(
            os.path.join(draw_img_save_dir, "system_results.txt"),
            'w',
            encoding='utf-8') as f:
        f.writelines(save_results)


class OCR:
    def __init__(self) -> None:
        pass
        args = utility.parse_args()
        from weight import config
        cfg = config.config 

        args.det_model_dir = cfg['det_model_dir']
        args.rec_model_dir =  cfg['rec_model_dir']
        args.rec_image_shape= cfg['rec_image_shape']
        args.rec_char_dict_path= cfg['rec_char_dict_path']
        args.rec_algorithm= cfg['rec_algorithm']
        args.rec_batch_num = cfg['rec_batch_num']
        args.use_openvino = cfg['use_openvino']

        self.PostProcess = cfg['imei_postprocessing']
        self.is_visualize = cfg['visualize']
        args.image_dir ="example/"
        args.image_dir = os.path.join(__dir__, args.image_dir)
        args.draw_img_save_dir = os.path.join(__dir__, args.draw_img_save_dir)
        args.vis_font_path = os.path.join(__dir__, args.vis_font_path)
        #args.draw_img_save_dir="inference_results"
        
        self.args = args
        self._init_model(args)

    def _init_model(self, args):
        
        
        self.text_sys = TextSystem(args)
        
        
        os.makedirs(args.draw_img_save_dir, exist_ok=True)
        

        logger.info(
            "In PP-OCRv3, rec_image_shape parameter defaults to '3, 48, 320', "
            "if you are using recognition model with PP-OCRv2 or an older version, please set --rec_image_shape='3,32,320"
        )

        self.preprocess = utility.My_PreProcess()
        # warm up 10 times
        if args.warmup:
            img = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
            for i in range(10):
                res = self.text_sys(img)
    def __call__(self, image=None, return_data=False, crop_already=True):
        
            if image is None:

                self.__call_image_file(crop_already=crop_already)
            else:
                return self.__call_image(image, return_data=return_data, crop_already=crop_already, PostProcess= self.PostProcess)

        

    def __call_image(self, image, return_data=False, crop_already=False, PostProcess=None, visualize=False):
	
            total_time = 0  
        
            if not crop_already:
                img = self.preprocess(image)
            else:
                img = image
            imgs = [img]
            
            for index, img in enumerate(imgs):
                starttime = time.time()
                dt_boxes, rec_res, time_dict = self.text_sys(img)
                elapse = time.time() - starttime
                total_time += elapse

                if len(imgs) > 1:
                    logger.debug(
                         str(index) + "  Predict time of images: %.3fs"
                        % ( elapse))
                else:
                    logger.debug(
                         "  Predict time of images: %.3fs" % elapse)
                for text, score in rec_res:
                    logger.debug("{}, {:.3f}".format(text, score))


                if return_data:
                    #image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    boxes = dt_boxes
                    txts = [rec_res[i][0] for i in range(len(rec_res))]
                    scores = [rec_res[i][1] for i in range(len(rec_res))]
                    #print(boxes)
                    
                    return boxes, txts, scores
                
                res = [{
                    "transcription": rec_res[i][0],
                    "points": np.array(dt_boxes[i]).astype(np.int32).tolist(),
                } for i in range(len(dt_boxes))]

                if PostProcess is not None:
                    text = PostProcess(res)
                else:
                    text = res
            

                
                image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                boxes = dt_boxes
                txts = [rec_res[i][0] for i in range(len(rec_res))]
                scores = [rec_res[i][1] for i in range(len(rec_res))]

                self.args.vis_font_path = os.path.join(__dir__, self.args.vis_font_path)
                draw_img = draw_ocr_box_txt(
                    image,
                    boxes,
                    txts,
                    scores,
                    drop_score=self.args.drop_score,
                    font_path=self.args.vis_font_path)
                
                if self.is_visualize:
                    save_file = 'image_%d' % len(os.listdir(self.args.draw_img_save_dir))
                    cv2.imwrite(
                        os.path.join(self.args.draw_img_save_dir,
                                    os.path.basename(save_file)),
                        draw_img[:, :, ::-1])
                    
                    logger.debug("The visualized image saved in {}".format(
                        os.path.join(self.args.draw_img_save_dir, os.path.basename(
                            save_file))))
                
                return draw_img, text


    def __call_image_file(self, crop_already=True):
	
        image_file_list, _ = get_image_file_list(self.args.image_dir)

        total_time = 0
        cpu_mem, gpu_mem, gpu_util = 0, 0, 0
        _st = time.time()
        count = 0
        #print(image_file_list)
        save_results = []
        for idx, image_file in enumerate(image_file_list):
            #print(image_file)
            if not crop_already:
                try:
                    img = self.preprocess(image_file)
                except:
                    img = cv2.imread(image_file)
            else:
                img = cv2.imread(image_file)
            imgs = [img]
            
            for index, img in enumerate(imgs):
                starttime = time.time()
                dt_boxes, rec_res, time_dict = self.text_sys(img)
                elapse = time.time() - starttime
                total_time += elapse
                if len(imgs) > 1:
                    logger.debug(
                        str(idx) + '_' + str(index) + "  Predict time of %s: %.3fs"
                        % (image_file, elapse))
                else:
                    logger.debug(
                        str(idx) + "  Predict time of %s: %.3fs" % (image_file,
                                                                    elapse))
                for text, score in rec_res:
                    logger.debug("{}, {:.3f}".format(text, score))

                res = [{
                    "transcription": rec_res[i][0],
                    "points": np.array(dt_boxes[i]).astype(np.int32).tolist(),
                } for i in range(len(dt_boxes))]

                if len(imgs) > 1:
                    save_pred = os.path.basename(image_file) + '_' + str(
                        index) + "\t" + json.dumps(
                            res, ensure_ascii=False) + "\n"
                else:
                    save_pred = os.path.basename(image_file) + "\t" + json.dumps(
                        res, ensure_ascii=False) + "\n"
                save_results.append(save_pred)

                if self.is_visualize:
                    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    boxes = dt_boxes
                    txts = [rec_res[i][0] for i in range(len(rec_res))]
                    scores = [rec_res[i][1] for i in range(len(rec_res))]

                    draw_img = draw_ocr_box_txt(
                        image,
                        boxes,
                        txts,
                        scores,
                        drop_score=self.args.drop_score,
                        font_path=self.args.vis_font_path)
                    
                    save_file = image_file
                    cv2.imwrite(
                        os.path.join(self.args.draw_img_save_dir,
                                    os.path.basename(save_file)),
                        draw_img[:, :, ::-1])
                    
                    logger.debug("The visualized image saved in {}".format(
                        os.path.join(self.args.draw_img_save_dir, os.path.basename(
                            save_file))))

        logger.info("The predict total time is {}".format(time.time() - _st))
        if self.args.benchmark:
            self.text_sys.text_detector.autolog.report()
            self.text_sys.text_recognizer.autolog.report()

        with open(
                os.path.join(self.args.draw_img_save_dir, "system_results.txt"),
                'w',
                encoding='utf-8') as f:
            f.writelines(save_results)



if __name__ == "__main__":
    args = utility.parse_args()
    print(os.getcwd())
    args.det_model_dir =  'weight/DB_det/inference.pdmodel'
    args.rec_model_dir ="weight/Rec/svtr_120k_endict/svtr_120k_endict.pdmodel"
    args.image_dir ="example/"
    args.rec_image_shape= "3, 64, 256"
    #args.rec_algorithm= "SVTR"
    #args.draw_img_save_dir="inference_results"
    #main(args)
    print( os.path.isdir(args.det_model_dir))
    ocr = OCR()
    ocr()


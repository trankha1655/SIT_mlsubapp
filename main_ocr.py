
from PaddleOCR.ocr import OCR 
import cv2
import sys
import os
import argparse
import json

root_image = "PaddleOCR/example"
image_files = ['mat_detect_bacode_failed.jpg', 'mat_detect_bacode_pased.jpg', 'mat_detect_bacode.jpg', 'test.jpg']

text_processing = OCR() 
#This OCR class with change os directory (to import module, model weight,...), 
#hence the below code gonna fix this. 
__dir__ = os.path.dirname(__file__)
os.environ['PATH'] =   __dir__
os.chdir("..")

parser = argparse.ArgumentParser(
description='Detect barcode from docker')
parser.add_argument('--image_path', required=True,
                metavar="/path/file/contain/list/path/images.jpg",
                help='Directory of the image detect')
parser.add_argument('--result_path', required=True,
                metavar="/path/file/result",
                help='Directory of the output result')
parser.add_argument('--debug_image', default=None, 
                help="Directory of OCR result image. If is folder, func will save as Folder + file_name, else save as image.png")
args = parser.parse_args()



if __name__=="__main__":
    image_files = [args.image_path]
    
    for file in image_files:

        #Read image
        image_file = os.path.join(root_image, file)
        image = cv2.imread(file)
        if image is None:
            print("Cannot read image file, please check path %s" %file)
            #continue

        #Prediction
        output_image, text = text_processing(image= image)

        #Save debug image
        if args.debug_image is not None:
            print(args.debug_image)
            if os.path.exists(args.debug_image): 
                image_file_write = os.path.join(args.debug_image, 'debug_image.png') if os.path.isdir(args.debug_image) else args.debug_image
                print("Saved debug image at ", image_file_write )
                cv2.imwrite(image_file_write, output_image)
        #Save output text
        f = open(args.result_path, 'w')
        if text != False and text is not None:
            if isinstance(text, str):
                f.write(text)
            elif isinstance(text, list):
                save_pred = os.path.basename(file) + "\t" + json.dumps(
                        text, ensure_ascii=False) + "\n"
                 
                f.write(save_pred)
            f.close()
        else:
            f.write('') 
            f.close()

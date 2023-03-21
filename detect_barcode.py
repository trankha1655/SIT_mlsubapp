import os
import sys
# import cv2
import json
from typing import List
from dbr import *
import cv2

import argparse
# Parse command line arguments
parser = argparse.ArgumentParser(
description='Detect barcode from docker')
parser.add_argument('--image_path', required=True,
                metavar="/path/file/contain/list/path/images.jpg",
                help='Directory of the image detect')
parser.add_argument('--result_path', required=True,
                metavar="/path/file/result",
                help='Directory of the mask detected')
args = parser.parse_args()

PATH_IMAGE = args.image_path
PATH_RESULT = args.result_path

# you can replace the following variables' value with yours.
license_key = "t0076xQAAADuZblMHxQg58GXQutgTiiKN/ATlr3XybdoANCzVqllT5N3owQ98Jp/F9lBip5ws9HLwlnspGgLq73cMIuwW1MzuWPcHJEgpoA=="
reader = BarcodeReader()
reader.init_license(license_key)


def get_location(points, image_src):
    height, width = image_src.shape[:2]

    min_x = width
    min_y = height
    max_x = 0
    max_y = 0
    for point in points:
        if point[0] < min_x:
            min_x = point[0]
        if point[1] < min_y:
            min_y = point[1]
        if point[0] > max_x:
            max_x = point[0]
        if point[1] > max_y:
            max_y = point[1]

    is_accept = True
    if width > 2000:
        max_x += 50
        max_y += 50
        min_x -= 50
        min_y -= 50
    else:
        max_x += 10
        max_y += 10
        min_x -= 10
        min_y -= 10

    if min_x < 0:
        min_x = 0
    if min_y < 0:
        min_y = 0

    if max_x >= width:
        max_x = width - 1
    if max_y >= height:
        max_y = height - 1

    if max_x - min_x <= 0:
        is_accept = False
    if max_y - min_y <= 0:
        is_accept = False   
    return is_accept, min_x, max_x, min_y, max_y

def edit_location(min_x, min_y, points):
    Point_new = []
    for point in points:
        point_tmp = (point[0]+min_x, point[1]+min_y)
        Point_new.append(point_tmp)
    return Point_new
           
def try_with_location(image_path, list_localization_points, barcodes):

    if(len(list_localization_points) > 0):
        image_src = cv2.imread(image_path, cv2.IMREAD_COLOR)
        basename = os.path.basename(image_path)
        filename, file_extension = os.path.splitext(basename)
        save_dir = os.path.dirname(image_path)
        index = 0
        for localization_points in  list_localization_points:

            is_accept, min_x, max_x, min_y, max_y = get_location(localization_points, image_src)

            if is_accept == True:
                index += 1
                image_crop = image_src[min_y:max_y, min_x:max_x]
                path_save = os.path.join(save_dir, '{}_bacode_crop_{}{}'.format(filename, index, file_extension))
                cv2.imwrite(path_save, image_crop)

                try:
                    print("path_save image_crop: {}".format(path_save))
                    text_results = reader.decode_file(path_save)
                    if text_results != None:
                        for text_result in text_results:
                            print("[Crop] Barcode Format : ")
                            print(text_result.barcode_format_string)
                            print("[Crop] Barcode Text : ")
                            print(text_result.barcode_text)
                            print("[Crop] Localization Points : ")
                            edit_localization_points = edit_location(min_x, min_y, text_result.localization_result.localization_points)
                            print(edit_localization_points)
                            # jsonBarcodeItem = {}
                            # jsonBarcodeItem["format"] = text_result.barcode_format_string
                            # jsonBarcodeItem["text"] = text_result.barcode_text
                            # jsonBarcodeItem["localization"] = edit_localization_points
                            # jsonBarcode.append(jsonBarcodeItem)
                            barcodes.append(text_result.barcode_text)

                except BarcodeReaderError as bre:
                    print(bre)

    return barcodes

def read_barcode(path_image, barcodes):
    try:
        text_results = reader.decode_file(path_image)
        print("text_results: ", text_results)
        if text_results != None:
            list_localization_points = []
            for text_result in text_results:
                print("Barcode Format : ")
                print(text_result.barcode_format_string)
                print("Barcode Text : ")
                print(text_result.barcode_text)
                print("Localization Points : ")
                print(text_result.localization_result.localization_points)
                print("Exception : ")
                print(text_result.exception)
                string  = str(text_result.barcode_bytes)
                string = string.replace("bytearray(b'","")
                string = string.replace("')","")
                print("string : ", string)
                print("-------------")
                barcodes.append(text_result.barcode_text)
                list_localization_points.append(text_result.localization_result.localization_points)
            barcodes = try_with_location(path_image, list_localization_points, barcodes)

    except BarcodeReaderError as bre:
        print(bre)
    return barcodes



if __name__ == '__main__':

    print(("PATH_IMAGE: ", PATH_IMAGE))
    print(("PATH_RESULT: ", PATH_RESULT))

    print ("-----------------------------")
    barcodes = []
    if '.png' in PATH_IMAGE or '.jpg' in PATH_IMAGE:
        print(("Running on {}".format(PATH_IMAGE.strip())))
        barcodes = read_barcode(PATH_IMAGE.strip(), barcodes)
    else:
        with open(PATH_IMAGE) as fp:
            line = fp.readline().strip()
            line = line.strip()
            while line:
                if (line.strip() != ''):
                    print(("Running on {}".format(line.strip())))
                    barcodes = read_barcode(line.strip(), barcodes)

                line = fp.readline()
                print(line)

    print(barcodes)
    with open(PATH_RESULT, 'w') as fp:
        fp.write('\n'.join(barcodes))
        print("Masks path saved to '%s'"% PATH_RESULT)
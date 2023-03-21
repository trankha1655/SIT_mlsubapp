```bash
$ROOT = "/home/greystone/SIT/"
#Barcode reader 

#Use can change to accomodate with your code 
$IMAGE_PATH = $ROOT + "PaddleOCR/example/mat_detect_bacode_failed.jpg"      #Required
$RESUT_PATH = $ROOT + "Result/barcode_result.txt"                           #Required
$FILE_SCRIPT= $ROOT + "detect_barcode.py"                                   #Required

# Create docker container from docker image 
sudo docker run -e LD_PRELOAD=/usr/local/lib/faketime/libfaketime.so.1 -e FAKETIME="2021-05-12 10:30:00" \
                    -it -d --network none --name docker_barcode \
                    -v /home/greystone:/home/greystone docker_barcode:set_time 

#Excute barcode
sudo docker exec docker_barcode python3.7 $FILE_SCRIPT --image_path=$IMAGE_PATH --result_path=$RESUT_PATH


#OCR reader (exec while Barcode could not recoginize)

$IMAGE_PATH = $ROOT + "PaddleOCR/example/mat_detect_bacode_failed.jpg"      #Required
$RESUT_PATH = $ROOT + "Result/ocr_result.txt"                               #Required
$FILE_SCRIPT= $ROOT + "main_ocr.py"                                         #Required
$DEBUG_IMAGE = $ROOT + "debug_image/"                                       #Optional

# Create docker container ppocr_openvino
sudo docker run --name ppocr_openvino -v /home/greystone:/home/greystone --shm-size=2g -it  paddleocr_openvino:v1.2 
#Execute OCR
sudo docker exec ppocr_openvino python3 $FILE_SCRIPT --image_path=$IMAGE_PATH --result_path=$RESUT_PATH --debug_image=$DEBUG_IMAGE
```
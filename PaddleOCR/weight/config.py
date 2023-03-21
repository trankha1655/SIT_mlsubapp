

from sit_postprocessing import SITPostProcessing
config = {
    'det_model_dir':        'weight/Det/ch_PP-OCRv3_det_infer/inference.pdmodel',
    'rec_model_dir' :       "weight/Rec/en_PP-OCRv3_rec_infer/inference.pdmodel",
    'rec_image_shape':      "3, 48, 320",
    'rec_batch_num':        4, 
    'rec_char_dict_path':   "ppocr/utils/en_dict.txt",
    'use_openvino':         True,
    'rec_algorithm':        "SVTR_LCNet",
    'use_clahe':            False,
    'imei_postprocessing':  None,
    'visualize':            False
}

# If use Open VINO, model dir must have fully suffix (.pdmodel or .onnx)
# Else: model dir doesn't need a name of file

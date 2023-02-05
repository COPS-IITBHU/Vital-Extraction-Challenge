from paddleocr import PaddleOCR,draw_ocr
DEVICE = "cpu"#"cuda" if torch.cuda.is_available() else "cpu"

# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `fr`, `german`, `korean`, `japan`
# to switch the language model in order.
ocr_m = PaddleOCR(lang='en', use_gpu=(DEVICE=="gpu"), det_db_box_thresh=0.6, drop_score = 0.4) # need to run only once to download and load model into memory

def get_text(crop_img, det=False):
    return ocr_m.ocr(crop_img, det=det)
    # for idx in range(len(result)):
    #     res = result[idx]
    #     for line in res:
    #         print(line)
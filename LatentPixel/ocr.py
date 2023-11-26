import torch
from paddleocr import PaddleOCR
from PIL.Image import Image
import pytesseract

import numpy as np

PADDLE: PaddleOCR | None = None

def init_paddle():
    global PADDLE
    PADDLE = PaddleOCR(det=False, cls = False, lang="en")

def ocr(imgs: list[Image] | Image, method: str, scale: float) -> list[str] | str:
    if isinstance(imgs, list):
        h = imgs[0].size[0]
        w = imgs[0].size[1]
        imgss = [img.resize((int(h*scale), int(w*scale))) for img in imgs]
    else:
        h = imgs.size[0]
        w = imgs.size[1]
        imgss = imgs.resize((int(h*scale), int(w*scale)))

    match method.lower():
        case 'paddleocr':
            ocr_fn = _paddleocr
        case 'tesseract':
            ocr_fn = _tesseract
        case _:
            raise KeyError(f'Wrong ocr method :{method}')
        
    return ocr_fn(imgss)

def _one_image_paddleocr(img: Image) -> str:
    global PADDLE
    if PADDLE is None:
        init_paddle()
    img = np.array(img)
    rec = PADDLE.ocr(img, cls=False)
    print(rec)
    result = ''
    if rec[0] is None:
        return ''
    for res in rec[0]:
        result += res[1][0]
    return result

def _paddleocr(imgs: list[Image] | Image) -> list[str] | str:
    if not isinstance(imgs, list):
        return _one_image_paddleocr(imgs)
    results = [_one_image_paddleocr(img) for img in imgs]
    return results

@torch.no_grad()
def _tesseract(imgs: list[Image] | Image) -> list[str] | str:
    if isinstance(imgs, list):
        text = [pytesseract.image_to_string(img) for img in imgs]
    else:
        text = pytesseract.image_to_string(imgs)
    return text

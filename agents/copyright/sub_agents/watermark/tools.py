import cv2
import os

from imwatermark import WatermarkEncoder, WatermarkDecoder

from google.adk.tools import ToolContext


def encode_image(image_path: str, metadata: str)-> str:
    try:
        image_path = os.path.abspath(image_path)
        
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        encoder = WatermarkEncoder()
        encoder.set_watermark('bytes', metadata.encode('utf-8'))
        bgr_encoded = encoder.encode(bgr, 'dwtDct')

        output_path = image_path.replace('.png', '_wm.png')
        success = cv2.imwrite(output_path, bgr_encoded)
        
        if not success:
            raise IOError("Failed to save image.")

        return output_path
    
    except Exception as e:
        return f"[encode_image error] {e}"
    
    
# 추후 수정 필요   
# def decode_image(image_path = PROTECTED_PATH):
#     print('[decode] 시작')
#     bgr = cv2.imread(image_path + '/ex1_wm.png')
#     if bgr is None:
#         raise ValueError('워터마크된 이미지 파일을 읽을 수 없습니다.')
    
#     watermark_bytes = WATERMARK.encode('utf-8')
#     decoder = WatermarkDecoder('bytes', len(watermark_bytes) * 8)
    
#     watermark = decoder.decode(bgr, 'dwtDct')

#     print('[decode] 완료')
#     print('raw watermark bytes:', watermark)
#     print('decoded watermark: ', watermark.decode('utf-8', errors='replace'))

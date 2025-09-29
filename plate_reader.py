# plate_reader.py
from ultralytics import YOLO
from paddleocr import PaddleOCR
import cv2
import numpy as np
import os

class PlateReader:
    def __init__(self, yolo_model_path: str, use_gpu: bool = False):
        # Load YOLO detection model
        self.det_model = YOLO(yolo_model_path)
        print(f"[INFO] YOLO model loaded from {yolo_model_path}")

        # Load PaddleOCR once
        self.ocr = PaddleOCR(
            use_angle_cls=False,
            lang='en',
            use_gpu=use_gpu,
            show_log=False
        )
        print("[INFO] PaddleOCR initialized")

    def read_plate(self, image_input):
        """
        image_input: str (path) or numpy array (BGR)
        Returns: list of dicts: [{text, ocr_conf, det_conf, bbox}]
        """
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError(f"Cannot read image: {image_input}")
        else:
            img = image_input

        H, W = img.shape[:2]
        results = []

        # 1) YOLO detect plates
        detections = self.det_model.predict(source=img, conf=0.25, save=False, verbose=False)

        for r in detections:
            if r.boxes is None or len(r.boxes) == 0:
                continue

            for i, box in enumerate(r.boxes.xyxy):
                x1, y1, x2, y2 = map(int, box.tolist())
                det_conf = float(r.boxes.conf[i])

                # Pad and crop
                pad = 5
                x1, y1 = max(0, x1-pad), max(0, y1-pad)
                x2, y2 = min(W, x2+pad), min(H, y2+pad)
                crop = img[y1:y2, x1:x2]

                # OCR
                ocr_result = self.ocr.ocr(crop, cls=False)
                if not ocr_result or not ocr_result[0]:
                    results.append({
                        "text": "",
                        "ocr_conf": 0.0,
                        "det_conf": det_conf,
                        "bbox": [x1, y1, x2, y2]
                    })
                    continue

                # take best line
                best = max(ocr_result[0], key=lambda x: x[1][1])
                text = best[1][0]
                ocr_conf = float(best[1][1])

                results.append({
                    "text": text,
                    "ocr_conf": ocr_conf,
                    "det_conf": det_conf,
                    "bbox": [x1, y1, x2, y2]
                })

        return results

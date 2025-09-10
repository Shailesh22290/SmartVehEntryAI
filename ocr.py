import cv2
from ultralytics import YOLO
import easyocr
import os

# Load models
det_model = YOLO("runs/detect/plate-detector/weights/best.pt")  # your trained YOLO model
reader = easyocr.Reader(['en'])  # OCR reader

# Input & output paths
input_path = "dataset/test/images"
output_path = "results"
os.makedirs(output_path, exist_ok=True)

# Run detection
results = det_model.predict(source=input_path, conf=0.5, save=False)

for r in results:
    img = cv2.imread(r.path)  # Original image
    for box in r.boxes.xyxy:  # Detected bounding boxes
        x1, y1, x2, y2 = map(int, box.tolist())
        crop = img[y1:y2, x1:x2]

        # OCR on cropped plate
        ocr_out = reader.readtext(crop)
        plate_text = ""
        if len(ocr_out) > 0:
            plate_text = ocr_out[0][1]  # Get text
            conf = ocr_out[0][2]
            print(f"{r.path} → Plate: {plate_text} (conf: {conf:.2f})")

            # Draw detection + OCR text on image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, plate_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Save output image
    save_path = os.path.join(output_path, os.path.basename(r.path))
    cv2.imwrite(save_path, img)

print(f"\n✅ Results saved in '{output_path}' folder")

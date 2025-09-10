# ocr_paddle.py
import os
import cv2
import argparse
import numpy as np
import csv
from ultralytics import YOLO
from paddleocr import PaddleOCR, draw_ocr
import paddle   # only used to check GPU availability (optional)

# ---------------------------
# Utilities: preprocessing variants
# ---------------------------
def make_variants(crop_bgr):
    """Return list of BGR numpy arrays: original + several enhanced variants."""
    variants = []
    orig = crop_bgr.copy()
    variants.append(orig)

    # convert to gray
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

    # CLAHE (good for uneven lighting)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    variants.append(cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR))

    # adaptive threshold (strong contrast)
    try:
        thr = cv2.adaptiveThreshold(gray, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
        variants.append(cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR))
    except Exception:
        pass

    # histogram equalization on Y channel (color)
    try:
        ycrcb = cv2.cvtColor(orig, cv2.COLOR_BGR2YCrCb)
        ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])
        he = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        variants.append(he)
    except Exception:
        pass

    # denoise
    try:
        den = cv2.fastNlMeansDenoisingColored(orig, None, 10, 10, 7, 21)
        variants.append(den)
    except Exception:
        pass

    # sharpen
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    sharp = cv2.filter2D(orig, -1, kernel)
    variants.append(sharp)

    # Resize variants to moderate size (optional): keep aspect ratio, set height=64..128
    processed = []
    for v in variants:
        h, w = v.shape[:2]
        # target height (OCR expects reasonably small height); adjust if needed
        target_h = 64
        scale = target_h / float(h) if h > 0 else 1.0
        new_w = max(32, int(w * scale))
        resized = cv2.resize(v, (new_w, target_h), interpolation=cv2.INTER_LINEAR)
        processed.append(resized)

    return processed

# ---------------------------
# Aggregation & normalization
# ---------------------------
import re
PLATE_RE = re.compile(r'^[A-Z]{2}\d{1,2}[A-Z]{1,2}\d{4}$')  # example Indian-style heuristic

def normalize_text(s):
    s = s.upper()
    s = re.sub(r'[^A-Z0-9]', '', s)
    return s

def score_candidate(text, ocr_conf, det_conf):
    # normalize then check plate regex heuristic
    n = normalize_text(text)
    boost = 1.0
    if PLATE_RE.match(n):
        boost = 1.3
    # combined score: product of detection confidence and OCR confidence, scaled by boost
    return float(det_conf) * float(ocr_conf) * boost, n

# ---------------------------
# Main pipeline
# ---------------------------
def process_all(yolo_path, input_path, output_dir, min_combined_conf=0.35, use_gpu='auto'):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "manual_review"), exist_ok=True)
    csv_out = os.path.join(output_dir, "ocr_results.csv")

    # 1) load detector
    det_model = YOLO(yolo_path)

    # 2) decide GPU for PaddleOCR
    paddle_gpu = False
    if use_gpu == 'auto':
        try:
            paddle_gpu = paddle.is_compiled_with_cuda()
        except Exception:
            paddle_gpu = False
    elif isinstance(use_gpu, bool):
        paddle_gpu = use_gpu

    # 3) load PaddleOCR
    # use_angle_cls=True helps rotated text (recommended for real world images).
    ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False, use_gpu=paddle_gpu)

    # 4) run detection on folder or single image
    results = det_model.predict(source=input_path, conf=0.3, save=False)  # lower det conf initially
    rows = []
    for r in results:
        img_path = r.path  # original image path
        orig_img = cv2.imread(img_path)
        if orig_img is None:
            print("Could not read", img_path)
            continue

        # If no boxes, save as no-detection
        if len(r.boxes) == 0:
            print(f"[NO DET] {img_path}")
            # optionally write copy with label
            no_out = orig_img.copy()
            cv2.putText(no_out, "NO_PLATE_DETECTED", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
            cv2.imwrite(os.path.join(output_dir, os.path.basename(img_path)), no_out)
            rows.append([img_path, "", 0.0, 0.0, "no_detection", ""])
            continue

        # iterate boxes (YOLO may find multiple plates)
        best_for_image = None

        for i, box in enumerate(r.boxes.xyxy):
            try:
                x1,y1,x2,y2 = map(int, box.tolist())
            except Exception:
                coords = box.cpu().numpy().astype(int)
                x1,y1,x2,y2 = coords[0],coords[1],coords[2],coords[3]

            # detection confidence (if available)
            try:
                det_conf = float(r.boxes.conf[i].cpu().numpy())
            except Exception:
                det_conf = 1.0

            # expand a little
            pad = 6
            H, W = orig_img.shape[:2]
            x1e = max(0, x1 - pad)
            y1e = max(0, y1 - pad)
            x2e = min(W, x2 + pad)
            y2e = min(H, y2 + pad)

            crop = orig_img[y1e:y2e, x1e:x2e]
            if crop.size == 0:
                continue

            # 5) build variants and OCR each
            variants = make_variants(crop)
            candidates = []  # (combined_score, normalized_text, ocr_conf, raw_text, variant_index)
            for vi, var in enumerate(variants):
                # PaddleOCR accepts numpy images (BGR or grayscale)
                try:
                    recs = ocr.ocr(var, cls=True)  # cls True uses angle classifier
                except Exception as ex:
                    # if paddle fails on a variant, skip
                    recs = []

                # recs is a list of [box_points, (text, conf)] items
                if not recs:
                    continue

                # choose the best OCR result (usually first) for this variant
                # but iterate through recs to be safe
                for rec in recs:
                    try:
                        txt = rec[1][0]
                        conf = float(rec[1][1])
                    except Exception:
                        # fallback attempt: some versions return different structure
                        if len(rec) >= 2:
                            txt = str(rec[1])
                            conf = 0.0
                        else:
                            continue
                    combined_score, normalized = score_candidate(txt, conf, det_conf)
                    candidates.append((combined_score, normalized, conf, txt, vi))

            # pick best candidate across variants
            if not candidates:
                # no OCR for this crop
                rows.append([img_path, "", 0.0, det_conf, "ocr_failed", f"{x1},{y1},{x2},{y2}"])
                continue

            candidates.sort(key=lambda x: x[0], reverse=True)
            best = candidates[0]  # best across variants

            combined_score, normalized, ocr_conf, raw_text, used_variant = best

            # If below threshold, mark for manual review and save the crop variant
            if combined_score < min_combined_conf:
                # save crop to manual_review for human labeling
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                crop_name = f"{base_name}_box{i}_var{used_variant}.jpg"
                cv2.imwrite(os.path.join(output_dir, "manual_review", crop_name), variants[used_variant])
                status = "low_confidence"
            else:
                status = "ok"

            # write annotated image (draw bbox and text)
            out_img = orig_img.copy()
            cv2.rectangle(out_img, (x1e,y1e), (x2e,y2e), (0,255,0), 2)
            label_text = f"{normalized} ({combined_score:.2f})"
            cv2.putText(out_img, label_text, (x1e, max(15,y1e-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            out_path = os.path.join(output_dir, os.path.basename(img_path))
            cv2.imwrite(out_path, out_img)

            # collect row
            rows.append([img_path, normalized, float(ocr_conf), float(det_conf), status, f"{x1e},{y1e},{x2e},{y2e}"])

    # write CSV
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path","plate_text","ocr_conf","det_conf","status","bbox"])
        writer.writerows(rows)

    print("Done. Results (annotated) saved in:", output_dir)
    print("CSV results:", csv_out)


# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo", required=True, help="path to trained YOLO .pt")
    parser.add_argument("--input", required=True, help="image file or folder")
    parser.add_argument("--output", default="results", help="output folder")
    parser.add_argument("--min_combined_conf", type=float, default=0.35)
    parser.add_argument("--use_gpu", default="auto", help="'auto' or True/False")
    args = parser.parse_args()

    # normalize use_gpu
    if args.use_gpu.lower() in ("true","1","t","yes","y"):
        use_gpu = True
    elif args.use_gpu.lower() in ("false","0","f","no","n"):
        use_gpu = False
    else:
        use_gpu = "auto"

    process_all(args.yolo, args.input, args.output, args.min_combined_conf, use_gpu)
# python ocr.py \
#   --yolo runs/detect/plate-detector/weights/best.pt \
#   --input dataset/test/images \
#   --output results \
#   --min_combined_conf 0.35 \
#   --use_gpu auto
# https://chatgpt.com/share/68c1bcd6-9050-800c-b40f-83d9d7211cd0
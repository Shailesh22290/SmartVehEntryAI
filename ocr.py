# ocr.py - Enhanced version with better error handling
import os
import cv2
import argparse
import numpy as np
import csv
from ultralytics import YOLO
from paddleocr import PaddleOCR
import paddle
import paddleocr  # For version access
import traceback

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
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
        variants.append(cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR))
    except Exception as e:
        print(f"CLAHE failed: {e}")

    # adaptive threshold (strong contrast)
    try:
        thr = cv2.adaptiveThreshold(gray, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
        variants.append(cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR))
    except Exception as e:
        print(f"Adaptive threshold failed: {e}")

    # histogram equalization on Y channel (color)
    try:
        ycrcb = cv2.cvtColor(orig, cv2.COLOR_BGR2YCrCb)
        ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])
        he = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        variants.append(he)
    except Exception as e:
        print(f"Histogram equalization failed: {e}")

    # denoise
    try:
        den = cv2.fastNlMeansDenoisingColored(orig, None, 10, 10, 7, 21)
        variants.append(den)
    except Exception as e:
        print(f"Denoising failed: {e}")

    # sharpen
    try:
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        sharp = cv2.filter2D(orig, -1, kernel)
        variants.append(sharp)
    except Exception as e:
        print(f"Sharpening failed: {e}")

    # Resize variants to moderate size
    processed = []
    for i, v in enumerate(variants):
        try:
            h, w = v.shape[:2]
            if h == 0 or w == 0:
                print(f"Skipping variant {i} - invalid dimensions: {h}x{w}")
                continue
                
            # target height for OCR
            target_h = 64
            scale = target_h / float(h)
            new_w = max(32, int(w * scale))
            
            if new_w > 0 and target_h > 0:
                resized = cv2.resize(v, (new_w, target_h), interpolation=cv2.INTER_LINEAR)
                processed.append(resized)
            else:
                print(f"Skipping variant {i} - invalid resize dimensions")
        except Exception as e:
            print(f"Resize failed for variant {i}: {e}")

    return processed

# ---------------------------
# Aggregation & normalization
# ---------------------------
import re
PLATE_RE = re.compile(r'^[A-Z]{2}\d{1,2}[A-Z]{1,2}\d{4}$')

def normalize_text(s):
    if not isinstance(s, str):
        s = str(s)
    s = s.upper()
    s = re.sub(r'[^A-Z0-9]', '', s)
    return s

def score_candidate(text, ocr_conf, det_conf):
    try:
        n = normalize_text(text)
        boost = 1.0
        if PLATE_RE.match(n):
            boost = 1.3
        return float(det_conf) * float(ocr_conf) * boost, n
    except Exception as e:
        print(f"Scoring failed: {e}")
        return 0.0, ""

# ---------------------------
# Safe OCR initialization
# ---------------------------
def initialize_ocr(use_gpu=False, max_retries=3):
    """Initialize PaddleOCR with multiple fallback strategies"""
    
    configs = [
        # Try with minimal config first
        {'use_angle_cls': False, 'lang': 'en', 'use_gpu': use_gpu, 'show_log': False},
        # Fallback to CPU if GPU fails
        {'use_angle_cls': False, 'lang': 'en', 'use_gpu': False, 'show_log': False},
        # Last resort - basic config
        {'lang': 'en', 'use_gpu': False},
    ]
    
    for attempt, config in enumerate(configs):
        try:
            print(f"Attempting OCR initialization (attempt {attempt + 1}) with config: {config}")
            ocr = PaddleOCR(**config)
            print(f"PaddleOCR initialized successfully with config: {config}")
            return ocr
        except Exception as e:
            print(f"OCR initialization attempt {attempt + 1} failed: {e}")
            if attempt == len(configs) - 1:
                print("All OCR initialization attempts failed")
                raise e
    
    return None

# ---------------------------
# Safe OCR processing
# ---------------------------
def safe_ocr_process(ocr, image_variant, max_retries=2):
    """Safely process OCR with error handling"""
    for attempt in range(max_retries):
        try:
            # Validate image
            if image_variant is None or image_variant.size == 0:
                return []
            
            h, w = image_variant.shape[:2]
            if h < 10 or w < 10:  # Too small for OCR
                return []
            
            # Try OCR
            results = ocr.ocr(image_variant, cls=False)  # Disable angle classifier for stability
            
            if results is None:
                return []
            
            # Handle nested list structure
            if isinstance(results, list) and len(results) > 0:
                if isinstance(results[0], list):
                    results = results[0]
            
            return results if results else []
            
        except Exception as e:
            print(f"OCR processing attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                return []
    
    return []

# ---------------------------
# Main pipeline
# ---------------------------
def process_all(yolo_path, input_path, output_dir, min_combined_conf=0.35, use_gpu='auto'):
    try:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "manual_review"), exist_ok=True)
        csv_out = os.path.join(output_dir, "ocr_results.csv")

        # 1) Load detector
        print("Loading YOLO model...")
        try:
            det_model = YOLO(yolo_path)
            print("YOLO model loaded successfully")
        except Exception as e:
            print(f"Failed to load YOLO model: {e}")
            return

        # 2) Determine GPU usage
        use_paddle_gpu = False
        if use_gpu == 'auto':
            try:
                use_paddle_gpu = paddle.is_compiled_with_cuda()
                print(f"Auto-detected GPU support: {use_paddle_gpu}")
            except Exception:
                use_paddle_gpu = False
                print("GPU detection failed, using CPU")
        elif isinstance(use_gpu, bool):
            use_paddle_gpu = use_gpu

        # 3) Initialize OCR
        print(f"Initializing PaddleOCR...")
        print(f"PaddlePaddle version: {paddle.__version__}")
        print(f"PaddleOCR version: {paddleocr.__version__}")
        
        try:
            ocr = initialize_ocr(use_paddle_gpu)
        except Exception as e:
            print(f"Failed to initialize OCR: {e}")
            return

        # 4) Run detection
        print("Running YOLO detection...")
        try:
            results = det_model.predict(source=input_path, conf=0.25, save=False, verbose=False)
        except Exception as e:
            print(f"YOLO prediction failed: {e}")
            return

        rows = []
        
        for r in results:
            try:
                img_path = r.path
                print(f"Processing: {img_path}")
                
                # Load image
                orig_img = cv2.imread(img_path)
                if orig_img is None:
                    print(f"Could not read image: {img_path}")
                    continue

                # Check for detections
                if r.boxes is None or len(r.boxes) == 0:
                    print(f"[NO DETECTION] {img_path}")
                    # Save image with "no detection" label
                    no_out = orig_img.copy()
                    cv2.putText(no_out, "NO_PLATE_DETECTED", (10,30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
                    out_path = os.path.join(output_dir, os.path.basename(img_path))
                    cv2.imwrite(out_path, no_out)
                    rows.append([img_path, "", 0.0, 0.0, "no_detection", ""])
                    continue

                # Process each detection
                best_result = None
                best_score = 0.0
                
                for i, box in enumerate(r.boxes.xyxy):
                    try:
                        # Get box coordinates
                        if hasattr(box, 'cpu'):
                            coords = box.cpu().numpy().astype(int)
                        else:
                            coords = np.array(box).astype(int)
                        
                        if len(coords) < 4:
                            continue
                            
                        x1, y1, x2, y2 = coords[:4]

                        # Get detection confidence
                        try:
                            if r.boxes.conf is not None and i < len(r.boxes.conf):
                                if hasattr(r.boxes.conf[i], 'cpu'):
                                    det_conf = float(r.boxes.conf[i].cpu().numpy())
                                else:
                                    det_conf = float(r.boxes.conf[i])
                            else:
                                det_conf = 0.5
                        except Exception:
                            det_conf = 0.5

                        # Expand bounding box
                        pad = 6
                        H, W = orig_img.shape[:2]
                        x1e = max(0, x1 - pad)
                        y1e = max(0, y1 - pad)
                        x2e = min(W, x2 + pad)
                        y2e = min(H, y2 + pad)

                        # Extract crop
                        if x2e <= x1e or y2e <= y1e:
                            print(f"Invalid bounding box for detection {i}")
                            continue
                            
                        crop = orig_img[y1e:y2e, x1e:x2e]
                        if crop.size == 0:
                            print(f"Empty crop for detection {i}")
                            continue

                        # Generate variants and perform OCR
                        variants = make_variants(crop)
                        if not variants:
                            print(f"No variants generated for detection {i}")
                            continue
                            
                        candidates = []
                        
                        for vi, var in enumerate(variants):
                            try:
                                ocr_results = safe_ocr_process(ocr, var)
                                
                                for rec in ocr_results:
                                    if rec is None:
                                        continue
                                        
                                    try:
                                        # Extract text and confidence
                                        if isinstance(rec, (list, tuple)) and len(rec) >= 2:
                                            text_info = rec[1]
                                            if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                                                txt = str(text_info[0]).strip()
                                                conf = float(text_info[1])
                                            else:
                                                txt = str(text_info).strip()
                                                conf = 0.5
                                        else:
                                            continue
                                            
                                        if len(txt) == 0:
                                            continue
                                            
                                        combined_score, normalized = score_candidate(txt, conf, det_conf)
                                        candidates.append((combined_score, normalized, conf, txt, vi))
                                        
                                    except Exception as e:
                                        print(f"Error processing OCR result: {e}")
                                        continue
                                        
                            except Exception as e:
                                print(f"OCR failed on variant {vi}: {e}")
                                continue

                        # Select best candidate
                        if not candidates:
                            print(f"No OCR results for detection {i}")
                            rows.append([img_path, "", 0.0, det_conf, "ocr_failed", f"{x1e},{y1e},{x2e},{y2e}"])
                            continue

                        candidates.sort(key=lambda x: x[0], reverse=True)
                        best = candidates[0]
                        combined_score, normalized, ocr_conf, raw_text, used_variant = best
                        
                        print(f"Best OCR result: '{normalized}' (score: {combined_score:.3f})")

                        # Determine status
                        if combined_score < min_combined_conf:
                            base_name = os.path.splitext(os.path.basename(img_path))[0]
                            crop_name = f"{base_name}_box{i}_var{used_variant}.jpg"
                            crop_path = os.path.join(output_dir, "manual_review", crop_name)
                            cv2.imwrite(crop_path, variants[used_variant])
                            status = "low_confidence"
                            print(f"Low confidence result saved: {crop_name}")
                        else:
                            status = "ok"

                        # Keep track of best result for this image
                        if combined_score > best_score:
                            best_score = combined_score
                            best_result = (normalized, ocr_conf, det_conf, status, x1e, y1e, x2e, y2e, combined_score)

                    except Exception as e:
                        print(f"Error processing detection {i}: {e}")
                        traceback.print_exc()
                        continue

                # Save annotated image and record result
                if best_result:
                    normalized, ocr_conf, det_conf, status, x1e, y1e, x2e, y2e, combined_score = best_result
                    
                    # Annotate image
                    out_img = orig_img.copy()
                    cv2.rectangle(out_img, (x1e, y1e), (x2e, y2e), (0, 255, 0), 2)
                    label_text = f"{normalized} ({combined_score:.2f})"
                    cv2.putText(out_img, label_text, (x1e, max(15, y1e-8)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    out_path = os.path.join(output_dir, os.path.basename(img_path))
                    cv2.imwrite(out_path, out_img)
                    
                    rows.append([img_path, normalized, float(ocr_conf), float(det_conf), status, f"{x1e},{y1e},{x2e},{y2e}"])
                else:
                    # No valid results
                    out_path = os.path.join(output_dir, os.path.basename(img_path))
                    cv2.imwrite(out_path, orig_img)
                    rows.append([img_path, "", 0.0, 0.0, "processing_failed", ""])

            except Exception as e:
                print(f"Error processing image {r.path}: {e}")
                traceback.print_exc()
                continue

        # Write CSV results
        try:
            with open(csv_out, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["image_path","plate_text","ocr_conf","det_conf","status","bbox"])
                writer.writerows(rows)
            print(f"CSV results saved: {csv_out}")
        except Exception as e:
            print(f"Failed to write CSV: {e}")

        print(f"\nProcessing complete!")
        print(f"Results saved in: {output_dir}")
        print(f"Manual review items in: {os.path.join(output_dir, 'manual_review')}")

    except Exception as e:
        print(f"Fatal error in main processing: {e}")
        traceback.print_exc()

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

    # Parse use_gpu argument
    if args.use_gpu.lower() in ("true","1","t","yes","y"):
        use_gpu = True
    elif args.use_gpu.lower() in ("false","0","f","no","n"):
        use_gpu = False
    else:
        use_gpu = "auto"

    process_all(args.yolo, args.input, args.output, args.min_combined_conf, use_gpu)

    
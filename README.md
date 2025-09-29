
# 🚗🔒 SmartVehEntryAI

An **AI-powered vehicle entry management system** that detects license plates with **YOLO** and extracts text using **PaddleOCR**.
Designed for **smart gates, parking systems, and access-controlled facilities**.

---

## ✨ Features

* 🔎 **YOLO-based plate detection** – custom-trained for high accuracy.
* 🔤 **Robust OCR with PaddleOCR** – supports multilingual plates.
* 🔄 **End-to-end pipeline:** Image/Video → Detect plate → OCR → Log result.
* 📷 Works with **images, videos, and live camera feeds**.
* ⚡ **FastAPI backend** for REST APIs and optional web dashboard.
* 🛡️ Handles **real-world conditions**: glare, skewed angles, low-light.

---

## 📂 Project Structure

```
SmartVehEntryAI/
│
├── app.py                  # FastAPI backend
├── ocr.py                  # Core detection + OCR pipeline
│
├── dataset/                # Training / testing data
│   ├── train/              # Training images & labels
│   ├── val/                # Validation images & labels
│   └── test/               # Test images & labels
│
├── runs/                   # YOLO training outputs (weights, logs)
├── scripts/                # Utility scripts (e.g., data prep)
├── results/                # Output folder for detections & OCR text
├── static/                 # Web UI static files (created at runtime)
├── requirements.txt        # Python dependencies
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone & create a virtual environment

```bash
git clone https://github.com/<your-username>/SmartVehEntryAI.git
cd SmartVehEntryAI

# Using venv
python3 -m venv veh_ai
source veh_ai/bin/activate        # Linux / macOS
# OR
veh_ai\Scripts\activate           # Windows (PowerShell)
```

### 2️⃣ Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> ✅ Make sure you have **Python ≥3.10** and **CUDA-enabled PyTorch** if using GPU.

---

## 🚀 Usage

### ▶️ 1. Run YOLO + OCR on images / videos

```bash
python ocr.py \
  --yolo  detection_model.pt \
  --input dataset/test/images \
  --output results \
  --min_combined_conf 0.35 \
  --use_gpu auto
```

**Arguments**:

* `--yolo` → Path to YOLO weights
* `--input` → Image(s) or video file / directory
* `--output` → Directory to save annotated images & OCR results
* `--min_combined_conf` → Detection + OCR confidence threshold
* `--use_gpu auto` → Auto-detect GPU (falls back to CPU)

---

### 🌐 2. Start the FastAPI Web Server

```bash
uvicorn app:app --reload
```

* Server runs at: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**
* Static assets served from `/static/`
* API endpoints: *(coming soon – to upload image, get OCR result)*

---

## 🖼️ Example Output

Input → YOLO plate detection → OCR text overlay

<p align="center">
  <img src="results/example_output.jpg" alt="Example output" width="500">
</p>

---

## 📊 Training (YOLO)

Retrain YOLO on your dataset:

```bash
yolo task=detect mode=train model=yolov8n.pt data=dataset.yaml epochs=50 imgsz=640
```

**dataset.yaml**

```yaml
train: dataset/train/images
val: dataset/val/images
nc: 1
names: ["number_plate"]
```

---

## 🛠️ Tech Stack

* [YOLOv8](https://github.com/ultralytics/ultralytics) – License plate detection
* [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) – OCR engine
* [FastAPI](https://fastapi.tiangolo.com/) – REST API & web backend
* Python 3.10+ • CUDA / CPU support

---

## 📈 Roadmap

* [ ] Add REST API endpoints for image upload & live camera feed
* [ ] Build a lightweight dashboard for viewing logs and results
* [ ] Integrate database for access-control records
* [ ] Add Dockerfile for easy deployment

---

## 📜 License

This project is released under the **MIT License**.
For commercial use or enterprise deployment, please contact the maintainers.


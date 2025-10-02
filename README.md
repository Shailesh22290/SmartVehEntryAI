# 🚗 SmartVehEntryAI

An **AI-powered vehicle entry management system** with dual interfaces for staff and administrators. Detects license plates with **YOLO**, extracts text using **PaddleOCR**, and provides comprehensive vehicle tracking with entry/exit management.

Designed for **smart gates, parking systems, security checkpoints, and access-controlled facilities**.

---

## Features

###  Features
*  **YOLO-based plate detection** – custom-trained for high accuracy
* **Robust OCR with PaddleOCR** – supports multilingual plates
*  **End-to-end pipeline:** Image/Video → Detect plate → OCR → Log result
*  Works with **images, videos, and live camera feeds**
*  Handles **real-world conditions**: glare, skewed angles, low-light

### User Interfaces
* 👨‍💼 **Staff Interface** (Port 8000)
  - Upload vehicle images or capture from camera
  - Real-time license plate detection and OCR
  - Form-based vehicle information entry
  - Driver name, vehicle type, and remarks logging
  - Auto-detection of entry/exit status

* 🔐 **Admin Panel** (Port 8001)
  - Password-protected dashboard
  - Real-time statistics and analytics
  - Interactive charts (vehicle types, entry/exit status)
  - Complete vehicle log management
  - Advanced filtering (status, date, vehicle number)
  - Edit and delete capabilities
  - CSV export for reports
  - Auto-refresh every 30 seconds

###  Database Features
* **SQLite database** for vehicle logs
*  Tracks: vehicle number, driver name, type, entry/exit times
*  Automatic entry/exit detection for repeat vehicles
*  Full search and filter capabilities

---

## 📂 Project Structure

```
SmartVehEntryAI/
│
├── app.py                  # Staff interface (FastAPI)
├── admin.py                # Admin panel (FastAPI)
├── database.py             # Database models and setup
├── plate_reader.py         # Core detection + OCR pipeline
│
├── detection_model.pt      # YOLO model weights
├── vehicles.db             # SQLite database (auto-created)
│
├── dataset/                # Training / testing data
│   ├── train/              # Training images & labels
│   ├── val/                # Validation images & labels
│   └── test/               # Test images & labels
│
├── runs/                   # YOLO training outputs (weights, logs)
├── scripts/                # Utility scripts (e.g., data prep)
├── static/                 # Annotated images (auto-created)
│
├── requirements.txt        # Python dependencies
├── start.bat               # Windows startup script
├── start.sh                # Linux/Mac startup script
└── README.md
```

---

##  Installation

### 1️ Clone & create a virtual environment

```bash
git clone https://github.com/<your-username>/SmartVehEntryAI.git
cd SmartVehEntryAI

conda create -n veh_ai python=3.10
conda activate veh_ai

```

### 2️ Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

>  **Requirements:** Python ≥3.10, CUDA-enabled PyTorch for GPU (optional)

---

##  Usage

###  Run Both Interfaces

#### **Windows:**
```bash
# Double-click or run:
start.bat
```

#### **Linux/Mac:**
```bash
chmod +x start.sh
./start.sh
```

###   Run Manually

Open **two terminal windows**:

**Terminal 1 - Staff Interface:**
```bash
conda activate veh_ai
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Admin Panel:**
```bash
conda activate veh_ai
uvicorn admin:admin --reload --host 0.0.0.0 --port 8001
```

---

##  Access the Application

| Interface | URL | Credentials |
|-----------|-----|-------------|
| **Staff Interface** | http://localhost:8000 | No login required |
| **Admin Panel** | http://localhost:8001 | Username: `admin`<br>Password: `admin123` |
| **Staff API Docs** | http://localhost:8000/docs | Auto-generated |
| **Admin API Docs** | http://localhost:8001/docs | Auto-generated |



## 📖 How It Works

### 1️ Staff Workflow
1. **Upload or capture** vehicle image
2. **Analyze** - AI detects license plate and extracts text
3. **Fill form** - Enter driver name, vehicle type, remarks
4. **Save** - Data logged to database with timestamp
5. **Auto-detection** - System recognizes if vehicle is entering or exiting

### 2️ Entry/Exit Logic
- **First scan** → Creates `ENTRY` record
- **Second scan** (same vehicle) → Updates record with `EXIT` time
- **Third scan** → Creates new `ENTRY` record (new visit)

### 3️ Admin Dashboard
- View real-time statistics
- Monitor all vehicle movements
- Edit incomplete records
- Generate reports
- Export data to CSV

---

## 🖼️ Example Output

### Staff Interface
- Clean, modern UI with Tailwind CSS
- Real-time camera capture
- Instant plate detection with confidence scores
- Responsive form for data entry

### Admin Dashboard
- Live statistics cards
- Interactive charts (Chart.js)
- Sortable, filterable data table
- Edit modal for quick updates

---

## 📊 Training (YOLO)

Retrain YOLO on your custom dataset:

```bash
yolo task=detect mode=train model=yolov8n.pt data=dataset.yaml epochs=50 imgsz=640
```

**dataset.yaml:**
```yaml
train: dataset/train/images
val: dataset/val/images
nc: 1
names: ["number_plate"]
```




## 🛠️ Tech Stack

### Backend
* [FastAPI](https://fastapi.tiangolo.com/) – Modern web framework
* [SQLAlchemy](https://www.sqlalchemy.org/) – ORM for database
* [Uvicorn](https://www.uvicorn.org/) – ASGI server

### AI/ML
* [YOLOv8](https://github.com/ultralytics/ultralytics) – License plate detection
* [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) – OCR engine
* [OpenCV](https://opencv.org/) – Image processing

### Frontend
* [Tailwind CSS](https://tailwindcss.com/) – Styling
* [Chart.js](https://www.chartjs.org/) – Data visualization
* [Material Icons](https://fonts.google.com/icons) – Icons
* Vanilla JavaScript – No framework overhead

### Database
* SQLite (default) – Easy setup, no configuration
* PostgreSQL (production) – Scalable, robust

---

## 📊 API Endpoints

### Staff Interface (`app.py`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Main upload interface |
| POST | `/predict` | Analyze vehicle image |
| POST | `/save-vehicle-info` | Save vehicle details |
| GET | `/health` | Health check |

### Admin Panel (`admin.py`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/login` | Login page |
| POST | `/login` | Authenticate admin |
| GET | `/logout` | Logout admin |
| GET | `/dashboard` | Admin dashboard |
| GET | `/api/vehicles` | Get all vehicles |
| PUT | `/api/vehicles/{id}` | Update vehicle |
| DELETE | `/api/vehicles/{id}` | Delete vehicle |
| GET | `/health` | Health check |

---


## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is released under the **MIT License**.

For commercial use or enterprise deployment, please contact the maintainers.

---

##  Acknowledgments

* YOLOv8 by Ultralytics
* PaddleOCR by PaddlePaddle
* FastAPI framework
* Tailwind CSS
* Chart.js

---

##  Support

For issues, questions, or contributions:
- 🐛 [Open an issue](https://github.com/shailesh22290/SmartVehEntryAI/issues)
- 💬 [Discussions](https://github.com/shailesh22290/SmartVehEntryAI/discussions)
- 📧 Email: shailesh22@iiserb.ac.in/shaileshkachhi786@gmail.com

---

<p align="center">Made with ❤️ for smarter, safer vehicle management</p>

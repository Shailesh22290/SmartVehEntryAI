from fastapi import FastAPI, UploadFile, File, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import uuid
import os
import logging
from io import BytesIO
from datetime import datetime, timedelta
from plate_reader import PlateReader
from fastapi import Depends
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from database import SessionLocal, VehicleLog, BannedVehicle

# Import admin app BEFORE creating main app
from admin import admin

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PlateVision Pro")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Init Plate Reader
try:
    pr = PlateReader("detection_model.pt", use_gpu=False)
    logger.info("Plate Reader initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Plate Reader: {e}")
    pr = None

# Ensure static dir exists
os.makedirs("static", exist_ok=True)

# Mount static files BEFORE mounting admin
app.mount("/static", StaticFiles(directory="static"), name="static")

# Mount admin app with proper prefix
app.mount("/admin", admin)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main page with upload, camera and form (all inline)."""
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>PlateVision Pro</title>
      <script src="https://cdn.tailwindcss.com"></script>
      <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
    </head>
    <body class="bg-gray-100 font-sans">

      <!-- Navbar -->
      <nav class="bg-white shadow p-4 sticky top-0 flex items-center justify-between">
        <div class="flex items-center gap-2">
          <span class="material-icons text-blue-600">camera_alt</span>
          <h1 class="text-xl font-semibold text-blue-600">PlateVision Pro</h1>
        </div>
        <a href="/admin/login" class="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 flex items-center gap-2">
          <span class="material-icons">admin_panel_settings</span>
          Admin Panel
        </a>
      </nav>

      <div class="container mx-auto p-6">

        <!-- Upload & Camera -->
        <div class="bg-white rounded-xl shadow p-6 mb-6">
          <h2 class="text-lg font-medium mb-2">Image Input</h2>
          <p class="text-gray-500 mb-4">Upload or capture a photo of a vehicle</p>

          <div class="flex flex-col gap-4 items-center">
            <!-- Upload -->
            <label for="fileInput"
              class="w-full border-2 border-dashed border-gray-400 rounded-lg p-6 text-center cursor-pointer hover:bg-blue-50">
              <span class="material-icons text-blue-500 text-5xl">cloud_upload</span>
              <p class="font-medium mt-2">Click or Drag Image Here</p>
              <p class="text-sm text-gray-500">PNG, JPG, WEBP up to 10MB</p>
              <input id="fileInput" type="file" accept="image/*" class="hidden">
            </label>

            <p class="text-gray-500 font-medium">OR</p>

            <!-- Camera -->
            <button id="cameraBtn"
              class="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 flex items-center gap-2">
              <span class="material-icons">photo_camera</span>
              Take Photo
            </button>

            <!-- Camera stream -->
            <div id="cameraContainer" class="hidden flex flex-col items-center gap-2">
              <video id="cameraStream" autoplay class="rounded-lg w-full max-h-64 bg-black"></video>
              <button id="captureBtn" class="bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700">
                Capture
              </button>
              <button id="closeCameraBtn" class="bg-gray-600 text-white px-4 py-2 rounded-lg hover:bg-gray-700">
                Close Camera
              </button>
            </div>
          </div>
        </div>

        <!-- Preview -->
        <div id="previewCard" class="bg-white rounded-xl shadow p-6 hidden">
          <img id="previewImg" class="rounded-lg w-full max-h-72 object-contain mb-4">
          <div class="flex gap-4 justify-center">
            <button id="analyzeBtn"
              class="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 flex items-center gap-2">
              <span class="material-icons">search</span>
              Analyze
            </button>
            <button id="clearBtn"
              class="bg-gray-200 px-4 py-2 rounded-lg hover:bg-gray-300 flex items-center gap-2">
              <span class="material-icons">clear</span>
              Clear
            </button>
          </div>
        </div>

        <!-- Results -->
        <div id="resultsCard" class="bg-white rounded-xl shadow p-6 hidden">
          <h3 class="text-lg font-semibold mb-4 flex items-center gap-2">
            <span class="material-icons text-green-600">check_circle</span>
            Analysis Complete
          </h3>
          <img id="resultImage" class="rounded-lg w-full mb-4">
          
          <!-- Status Badge -->
          <div id="statusBadge" class="mb-4"></div>
          
          <div id="plateList" class="space-y-3"></div>

          <!-- Vehicle Form -->
          <form id="vehicleForm" class="mt-6 space-y-4">
            <input type="hidden" id="entryTimeHidden" name="entry_time">
            
            <div>
              <label class="block text-sm font-medium">Vehicle Number *</label>
              <input id="vehicleNumber" name="vehicle_number" type="text" 
                     class="w-full border rounded-lg p-2 bg-gray-50" readonly required>
            </div>

            <div>
              <label class="block text-sm font-medium">Entry Time</label>
              <input id="entryTimeDisplay" type="text" class="w-full border rounded-lg p-2 bg-gray-50" readonly>
            </div>

            <div id="exitTimeContainer" class="hidden">
              <label class="block text-sm font-medium">Exit Time</label>
              <input id="exitTimeDisplay" type="text" class="w-full border rounded-lg p-2 bg-gray-50" readonly>
            </div>

            <div>
              <label class="block text-sm font-medium">Driver Name *</label>
              <input id="driverName" name="driver_name" type="text" 
                     class="w-full border rounded-lg p-2" 
                     placeholder="Enter driver name" required>
            </div>

            <div>
              <label class="block text-sm font-medium">Vehicle Type *</label>
              <select id="vehicleType" name="vehicle_type" class="w-full border rounded-lg p-2" required>
                <option value="">Select vehicle type</option>
                <option>Car</option>
                <option>Bus</option>
                <option>Truck</option>
                <option>Auto</option>
                <option>Two-Wheeler</option>
                <option>Other</option>
              </select>
            </div>

            <div>
              <label class="block text-sm font-medium">Remarks</label>
              <textarea id="remarks" name="remarks" rows="2" 
                        class="w-full border rounded-lg p-2" 
                        placeholder="Optional notes..."></textarea>
            </div>

            <button type="submit" id="saveBtn"
              class="bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 w-full flex items-center justify-center gap-2">
              <span class="material-icons">save</span>
              Save Vehicle Info
            </button>
          </form>
        </div>

        <!-- Success Message -->
        <div id="successCard" class="hidden bg-green-600 text-white rounded-lg p-4 mt-4 flex items-center gap-2">
          <span class="material-icons">check_circle</span>
          <span id="successMessage"></span>
        </div>

        <!-- Error -->
        <div id="errorCard" class="hidden bg-red-600 text-white rounded-lg p-4 mt-4 flex items-center gap-2">
          <span class="material-icons">error</span>
          <span id="errorMessage"></span>
        </div>

        <!-- Loading Overlay -->
        <div id="loadingOverlay" class="hidden fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div class="bg-white rounded-lg p-6 flex flex-col items-center gap-4">
            <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
            <p class="text-gray-700">Processing...</p>
          </div>
        </div>
      </div>

      <!-- JS -->
      <script>
        const fileInput = document.getElementById('fileInput');
        const previewCard = document.getElementById('previewCard');
        const previewImg = document.getElementById('previewImg');
        const analyzeBtn = document.getElementById('analyzeBtn');
        const clearBtn = document.getElementById('clearBtn');
        const errorCard = document.getElementById('errorCard');
        const errorMessage = document.getElementById('errorMessage');
        const successCard = document.getElementById('successCard');
        const successMessage = document.getElementById('successMessage');
        const resultsCard = document.getElementById('resultsCard');
        const resultImage = document.getElementById('resultImage');
        const plateList = document.getElementById('plateList');
        const vehicleNumber = document.getElementById('vehicleNumber');
        const entryTimeDisplay = document.getElementById('entryTimeDisplay');
        const entryTimeHidden = document.getElementById('entryTimeHidden');
        const exitTimeContainer = document.getElementById('exitTimeContainer');
        const exitTimeDisplay = document.getElementById('exitTimeDisplay');
        const statusBadge = document.getElementById('statusBadge');
        const loadingOverlay = document.getElementById('loadingOverlay');
        const vehicleForm = document.getElementById('vehicleForm');
        const saveBtn = document.getElementById('saveBtn');
        const driverName = document.getElementById('driverName');
        const vehicleType = document.getElementById('vehicleType');
        const remarks = document.getElementById('remarks');

        let selectedFile = null;
        let currentStream = null;
        let currentEntryTime = null;

        // Upload
        fileInput.addEventListener('change', e => {
          if (e.target.files.length > 0) handleFile(e.target.files[0]);
        });

        // Camera open
        cameraBtn.addEventListener('click', async () => {
          try {
            cameraContainer.classList.remove('hidden');
            currentStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
            cameraStream.srcObject = currentStream;
          } catch (err) {
            showError('Camera access denied: ' + err.message);
            cameraContainer.classList.add('hidden');
          }
        });

        // Close camera
        closeCameraBtn.addEventListener('click', () => {
          if (currentStream) {
            currentStream.getTracks().forEach(track => track.stop());
            currentStream = null;
          }
          cameraContainer.classList.add('hidden');
        });

        // Capture
        captureBtn.addEventListener('click', () => {
          const canvas = document.createElement('canvas');
          canvas.width = cameraStream.videoWidth;
          canvas.height = cameraStream.videoHeight;
          canvas.getContext('2d').drawImage(cameraStream, 0, 0);
          canvas.toBlob(blob => {
            handleFile(new File([blob], "capture.jpg", { type: "image/jpeg" }));
            closeCameraBtn.click();
          });
        });

        function handleFile(file) {
          if (!file.type.startsWith('image/')) { 
            showError("Invalid file type. Please upload an image."); 
            return; 
          }
          if (file.size > 10 * 1024 * 1024) {
            showError("File too large. Maximum size is 10MB.");
            return;
          }
          selectedFile = file;
          const reader = new FileReader();
          reader.onload = e => {
            previewImg.src = e.target.result;
            previewCard.classList.remove('hidden');
            hideMessages();
          };
          reader.readAsDataURL(file);
        }

        clearBtn.addEventListener('click', resetUI);
        
        function resetUI() {
          selectedFile = null;
          currentEntryTime = null;
          fileInput.value = '';
          previewCard.classList.add('hidden');
          resultsCard.classList.add('hidden');
          hideMessages();
          vehicleForm.reset();
          driverName.readOnly = false;
          vehicleType.disabled = false;
          remarks.readOnly = false;
          saveBtn.classList.remove('hidden');
          vehicleForm.classList.remove('pointer-events-none');
          exitTimeContainer.classList.add('hidden');
        }

        // Analyze
        analyzeBtn.addEventListener('click', async () => {
          if (!selectedFile) return;
          
          showLoading(true);
          const formData = new FormData();
          formData.append('file', selectedFile);
          
          try {
            const res = await fetch('/predict', { method: 'POST', body: formData });
            const data = await res.json();
            
            if (!res.ok) throw new Error(data.detail || 'Analysis failed');
            
            displayResults(data);
            hideMessages();
            if (data.status === 'EXIT') {
              showSuccess(`Vehicle ${data.vehicle_number} exited successfully!`);
            }
          } catch (err) { 
            showError(err.message); 
          } finally {
            showLoading(false);
          }
        });

        function displayResults(data) {
          resultImage.src = data.annotated_image_url;
          resultsCard.classList.remove('hidden');
          
          const status = data.status || 'ENTRY';
          const badgeColor = status === 'ENTRY' ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800';
          statusBadge.innerHTML = `
            <span class="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${badgeColor}">
              <span class="material-icons text-sm mr-1">${status === 'ENTRY' ? 'login' : 'logout'}</span>
              ${status}
            </span>
          `;
          
          plateList.innerHTML = '';
          
          if (data.results && data.results.length > 0) {
            const plate = data.results[0].text || "UNREADABLE";
            vehicleNumber.value = plate;
            
            currentEntryTime = data.entry_time || new Date().toISOString();
            entryTimeDisplay.value = new Date(currentEntryTime).toLocaleString();
            entryTimeHidden.value = currentEntryTime;
            
            if (status === 'EXIT') {
              driverName.value = data.driver_name || 'N/A';
              vehicleType.value = data.vehicle_type || '';
              remarks.value = data.remarks || '';
              driverName.readOnly = true;
              vehicleType.disabled = true;
              remarks.readOnly = true;
              saveBtn.classList.add('hidden');
              vehicleForm.classList.add('pointer-events-none');
              exitTimeContainer.classList.remove('hidden');
              exitTimeDisplay.value = data.exit_time ? new Date(data.exit_time).toLocaleString() : new Date().toLocaleString();
            } else {
              driverName.readOnly = false;
              vehicleType.disabled = false;
              remarks.readOnly = false;
              saveBtn.classList.remove('hidden');
              vehicleForm.classList.remove('pointer-events-none');
              driverName.value = '';
              vehicleType.value = '';
              remarks.value = '';
              exitTimeContainer.classList.add('hidden');
            }
            
            data.results.forEach(p => {
              const div = document.createElement('div');
              div.className = "p-3 bg-gray-100 rounded-lg flex justify-between items-center";
              div.innerHTML = `
                <span class="font-mono text-lg font-semibold">${p.text || 'N/A'}</span>
                <span class="text-sm text-green-600 font-medium">
                  Confidence: ${((p.confidence || 0) * 100).toFixed(1)}%
                </span>
              `;
              plateList.appendChild(div);
            });
          } else {
            plateList.innerHTML = '<p class="text-gray-500 text-center py-4">No license plates detected</p>';
            vehicleNumber.value = "UNREADABLE";
            currentEntryTime = new Date().toISOString();
            entryTimeDisplay.value = new Date().toLocaleString();
            entryTimeHidden.value = currentEntryTime;
            
            if (status === 'EXIT') {
              driverName.value = data.driver_name || 'N/A';
              vehicleType.value = data.vehicle_type || '';
              remarks.value = data.remarks || '';
              driverName.readOnly = true;
              vehicleType.disabled = true;
              remarks.readOnly = true;
              saveBtn.classList.add('hidden');
              vehicleForm.classList.add('pointer-events-none');
              exitTimeContainer.classList.remove('hidden');
              exitTimeDisplay.value = data.exit_time ? new Date(data.exit_time).toLocaleString() : new Date().toLocaleString();
            } else {
              driverName.readOnly = false;
              vehicleType.disabled = false;
              remarks.readOnly = false;
              saveBtn.classList.remove('hidden');
              vehicleForm.classList.remove('pointer-events-none');
              driverName.value = '';
              vehicleType.value = '';
              remarks.value = '';
              exitTimeContainer.classList.add('hidden');
            }
          }
        }

        vehicleForm.addEventListener('submit', async (e) => {
          e.preventDefault();

          const vNumber = vehicleNumber.value.trim();
          const dName = driverName.value.trim();
          const vType = vehicleType.value;

          if (!vNumber) {
            showError('Vehicle number is required');
            return;
          }
          if (!dName) {
            showError('Driver name is required');
            return;
          }
          if (!vType) {
            showError('Vehicle type is required');
            return;
          }

          showLoading(true);
          const formData = new FormData(vehicleForm);
          
          try {
            const res = await fetch('/save-vehicle-info', {
              method: 'POST',
              body: formData
            });
            const data = await res.json();
            
            if (res.ok) {
              showSuccess(data.message || "Vehicle info saved successfully!");
              setTimeout(() => {
                resetUI();
              }, 2000);
            } else {
              showError(data.detail || "Failed to save vehicle info");
            }
          } catch (err) {
            showError("Network error: " + err.message);
          } finally {
            showLoading(false);
          }
        });

        function showError(msg) {
          errorMessage.textContent = msg;
          errorCard.classList.remove('hidden');
          successCard.classList.add('hidden');
          setTimeout(() => errorCard.classList.add('hidden'), 5000);
        }

        function showSuccess(msg) {
          successMessage.textContent = msg;
          successCard.classList.remove('hidden');
          errorCard.classList.add('hidden');
          setTimeout(() => successCard.classList.add('hidden'), 5000);
        }

        function hideMessages() {
          errorCard.classList.add('hidden');
          successCard.classList.add('hidden');
        }

        function showLoading(show) {
          if (show) {
            loadingOverlay.classList.remove('hidden');
          } else {
            loadingOverlay.classList.add('hidden');
          }
        }
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.post("/predict")
async def predict(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Predict license plate from uploaded image."""
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        img_bytes = await file.read()
        if len(img_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
            
        npimg = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid or corrupted image")

        if pr is None:
            raise HTTPException(status_code=500, detail="Plate reader not initialized")

        results = pr.read_plate(img)
        logger.info(f"Detected {len(results)} plates")

        for det in results:
            x1, y1, x2, y2 = det["bbox"]
            text = det.get("text", "")
            conf = det.get("ocr_conf", 0)
            det_conf = det.get("det_conf", 0)
            combined = conf * det_conf
            det["confidence"] = combined
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{text} ({combined*100:.1f}%)", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        filename = f"{uuid.uuid4().hex}.jpg"
        save_path = os.path.join("static", filename)
        cv2.imwrite(save_path, img)

        vehicle_number = results[0]["text"].upper() if results and results[0].get("text") else "UNREADABLE"

        banned = db.query(BannedVehicle).filter(BannedVehicle.vehicle_number == vehicle_number).first()
        if banned:
            raise HTTPException(status_code=403, detail=f"Vehicle {vehicle_number} is banned: {banned.reason}")

        response_data = {
            "results": results,
            "annotated_image_url": f"/static/{filename}",
            "status": "ENTRY",
            "vehicle_number": vehicle_number,
            "driver_name": None,
            "vehicle_type": None,
            "remarks": None,
            "entry_time": datetime.now().isoformat(),
            "exit_time": None
        }

        try:
            existing_entry = db.query(VehicleLog).filter(
                VehicleLog.vehicle_number == vehicle_number,
                VehicleLog.exit_time == None
            ).first()

            if existing_entry:
                exit_time = datetime.now()
                existing_entry.exit_time = exit_time
                existing_entry.status = "EXIT"
                db.commit()
                response_data["status"] = "EXIT"
                response_data["driver_name"] = existing_entry.driver_name
                response_data["vehicle_type"] = existing_entry.vehicle_type
                response_data["remarks"] = existing_entry.remarks
                response_data["entry_time"] = existing_entry.entry_time.isoformat()
                response_data["exit_time"] = exit_time.isoformat()
                logger.info(f"Vehicle {vehicle_number} marked as EXIT")
            else:
                recent_exit = db.query(VehicleLog).filter(
                    VehicleLog.vehicle_number == vehicle_number,
                    VehicleLog.exit_time != None,
                    VehicleLog.exit_time >= datetime.now() - timedelta(minutes=5)
                ).first()

                if recent_exit:
                    response_data["status"] = "EXIT"
                    response_data["driver_name"] = recent_exit.driver_name
                    response_data["vehicle_type"] = recent_exit.vehicle_type
                    response_data["remarks"] = recent_exit.remarks
                    response_data["entry_time"] = recent_exit.entry_time.isoformat()
                    response_data["exit_time"] = recent_exit.exit_time.isoformat()
                    logger.info(f"Vehicle {vehicle_number} recently exited; no new entry created")
                else:
                    log = VehicleLog(
                        vehicle_number=vehicle_number,
                        entry_time=datetime.now(),
                        status="ENTRY",
                        image_path=save_path
                    )
                    db.add(log)
                    db.commit()
                    db.refresh(log)
                    response_data["status"] = "ENTRY"
                    logger.info(f"New vehicle entry: {vehicle_number}")

        except SQLAlchemyError as e:
            logger.error(f"Database error: {e}")
            db.rollback()
            response_data["status"] = "ENTRY"

        return JSONResponse(response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/save-vehicle-info")
async def save_vehicle_info(
    vehicle_number: str = Form(...),
    driver_name: str = Form(...),
    vehicle_type: str = Form(...),
    entry_time: str = Form(None),
    remarks: str = Form(""),
    db: Session = Depends(get_db)
):
    """Save additional vehicle information to database."""
    try:
        if not vehicle_number or not vehicle_number.strip():
            raise HTTPException(status_code=400, detail="Vehicle number is required")
        
        if not driver_name or not driver_name.strip():
            raise HTTPException(status_code=400, detail="Driver name is required")
        
        if not vehicle_type or not vehicle_type.strip():
            raise HTTPException(status_code=400, detail="Vehicle type is required")

        vehicle_number = vehicle_number.strip().upper()
        driver_name = driver_name.strip()
        vehicle_type = vehicle_type.strip()

        record = db.query(VehicleLog).filter(
            VehicleLog.vehicle_number == vehicle_number,
            VehicleLog.exit_time == None
        ).order_by(VehicleLog.entry_time.desc()).first()

        if record:
            record.driver_name = driver_name
            record.vehicle_type = vehicle_type
            record.remarks = remarks
            db.commit()
            db.refresh(record)
            
            logger.info(f"Updated vehicle info for {vehicle_number}")
            return JSONResponse({
                "message": "Vehicle information saved successfully",
                "id": record.id,
                "vehicle_number": vehicle_number
            })
        else:
            new_entry = VehicleLog(
                vehicle_number=vehicle_number,
                driver_name=driver_name,
                vehicle_type=vehicle_type,
                remarks=remarks,
                entry_time=datetime.now(),
                status="ENTRY"
            )
            db.add(new_entry)
            db.commit()
            db.refresh(new_entry)
            
            logger.info(f"Created new entry for {vehicle_number}")
            return JSONResponse({
                "message": "New vehicle entry created successfully",
                "id": new_entry.id,
                "vehicle_number": vehicle_number
            })
            
    except HTTPException:
        raise
    except SQLAlchemyError as e:
        logger.error(f"Database error in save_vehicle_info: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail="Database error occurred")
    except Exception as e:
        logger.error(f"Error in save_vehicle_info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to save: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/vehicles")
async def get_vehicles(db: Session = Depends(get_db)):
    """Get all vehicle logs."""
    try:
        vehicles = db.query(VehicleLog).order_by(VehicleLog.entry_time.desc()).limit(100).all()
        return [{
            "id": v.id,
            "vehicle_number": v.vehicle_number,
            "driver_name": v.driver_name,
            "vehicle_type": v.vehicle_type,
            "entry_time": v.entry_time.isoformat() if v.entry_time else None,
            "exit_time": v.exit_time.isoformat() if v.exit_time else None,
            "status": v.status,
            "remarks": v.remarks
        } for v in vehicles]
    except Exception as e:
        logger.error(f"Error fetching vehicles: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch vehicles")
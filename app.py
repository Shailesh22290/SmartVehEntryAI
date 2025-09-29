from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
from io import BytesIO
import uuid, os
from plate_reader import PlateReader

app = FastAPI(title="License Plate Reader API")

# Make sure static and templates folders exist
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Initialize PlateReader (same path as your working setup)
pr = PlateReader("detection_model.pt", use_gpu=False)

# ---------- Routes ---------- #

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Enhanced upload form with modern interface."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>PlateVision Pro</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
        <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
        <style>
            :root {
                --primary: #1976d2;
                --primary-dark: #0d47a1;
                --surface: #ffffff;
                --surface-variant: #f5f5f5;
                --on-surface: #1c1b1f;
                --on-surface-variant: #49454f;
                --outline: #79747e;
                --shadow: rgba(0, 0, 0, 0.12);
                --success: #4caf50;
                --error: #f44336;
                --elevation-1: 0 1px 3px var(--shadow);
                --elevation-2: 0 2px 6px var(--shadow);
                --elevation-3: 0 4px 12px var(--shadow);
            }

            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                background: #fafafa;
                color: var(--on-surface);
                line-height: 1.5;
            }

            .app-bar {
                background: var(--surface);
                box-shadow: var(--elevation-2);
                padding: 16px 24px;
                position: sticky;
                top: 0;
                z-index: 100;
            }

            .app-bar h1 {
                font-size: 20px;
                font-weight: 500;
                color: var(--primary);
                display: flex;
                align-items: center;
                gap: 8px;
            }

            .container {
                max-width: 800px;
                margin: 0 auto;
                padding: 24px;
            }

            .upload-card {
                background: var(--surface);
                border-radius: 12px;
                box-shadow: var(--elevation-1);
                margin-bottom: 24px;
                overflow: hidden;
                transition: box-shadow 0.2s ease;
            }

            .upload-card:hover {
                box-shadow: var(--elevation-2);
            }

            .card-header {
                padding: 20px 24px;
                border-bottom: 1px solid var(--outline);
            }

            .card-title {
                font-size: 16px;
                font-weight: 500;
                margin-bottom: 4px;
            }

            .card-subtitle {
                color: var(--on-surface-variant);
                font-size: 14px;
            }

            .upload-zone {
                padding: 48px 24px;
                text-align: center;
                border: 2px dashed var(--outline);
                margin: 24px;
                border-radius: 8px;
                cursor: pointer;
                transition: all 0.2s ease;
                background: var(--surface-variant);
            }

            .upload-zone:hover {
                border-color: var(--primary);
                background: #e3f2fd;
            }

            .upload-zone.dragover {
                border-color: var(--primary);
                background: #e3f2fd;
                transform: scale(1.01);
            }

            .upload-icon {
                color: var(--primary);
                font-size: 48px;
                margin-bottom: 16px;
            }

            .upload-text {
                font-size: 16px;
                font-weight: 500;
                margin-bottom: 8px;
            }

            .upload-hint {
                color: var(--on-surface-variant);
                font-size: 14px;
            }

            #fileInput {
                display: none;
            }

            .btn {
                background: var(--primary);
                color: white;
                border: none;
                border-radius: 20px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: 500;
                cursor: pointer;
                display: inline-flex;
                align-items: center;
                gap: 8px;
                margin-top: 16px;
                transition: all 0.2s ease;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }

            .btn:hover {
                background: var(--primary-dark);
                box-shadow: var(--elevation-2);
            }

            .btn:disabled {
                background: var(--outline);
                cursor: not-allowed;
            }

            .btn-secondary {
                background: transparent;
                color: var(--primary);
                border: 1px solid var(--primary);
            }

            .btn-secondary:hover {
                background: #e3f2fd;
            }

            .preview-card {
                background: var(--surface);
                border-radius: 12px;
                box-shadow: var(--elevation-1);
                padding: 24px;
                margin-bottom: 24px;
                display: none;
            }

            .preview-img {
                width: 100%;
                max-height: 300px;
                object-fit: contain;
                border-radius: 8px;
                margin-bottom: 16px;
            }

            .loading-card {
                background: var(--surface);
                border-radius: 12px;
                box-shadow: var(--elevation-1);
                padding: 48px 24px;
                text-align: center;
                display: none;
            }

            .progress-circle {
                width: 56px;
                height: 56px;
                margin: 0 auto 24px;
                position: relative;
            }

            .progress-circle::after {
                content: '';
                width: 100%;
                height: 100%;
                border: 4px solid #e3f2fd;
                border-top: 4px solid var(--primary);
                border-radius: 50%;
                position: absolute;
                animation: spin 1s linear infinite;
            }

            @keyframes spin {
                to { transform: rotate(360deg); }
            }

            .results-card {
                background: var(--surface);
                border-radius: 12px;
                box-shadow: var(--elevation-1);
                overflow: hidden;
                display: none;
            }

            .result-image {
                width: 100%;
                height: auto;
                display: block;
            }

            .results-content {
                padding: 24px;
            }

            .results-header {
                display: flex;
                align-items: center;
                gap: 8px;
                margin-bottom: 16px;
            }

            .results-title {
                font-size: 18px;
                font-weight: 500;
            }

            .chip {
                background: var(--primary);
                color: white;
                padding: 4px 12px;
                border-radius: 16px;
                font-size: 12px;
                font-weight: 500;
            }

            .plate-list {
                display: flex;
                flex-direction: column;
                gap: 12px;
            }

            .plate-item {
                background: var(--surface-variant);
                border-radius: 8px;
                padding: 16px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                transition: background 0.2s ease;
            }

            .plate-item:hover {
                background: #e8f5e8;
            }

            .plate-text {
                font-family: 'Monaco', 'Menlo', monospace;
                font-size: 18px;
                font-weight: 600;
                letter-spacing: 2px;
            }

            .confidence-badge {
                background: var(--success);
                color: white;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 12px;
                font-weight: 500;
            }

            .error-card {
                background: var(--error);
                color: white;
                border-radius: 8px;
                padding: 16px;
                margin-top: 16px;
                display: none;
            }

            .empty-state {
                text-align: center;
                padding: 32px;
                color: var(--on-surface-variant);
            }

            .fab {
                position: fixed;
                bottom: 24px;
                right: 24px;
                width: 56px;
                height: 56px;
                border-radius: 50%;
                background: var(--primary);
                color: white;
                border: none;
                cursor: pointer;
                box-shadow: var(--elevation-3);
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.2s ease;
                z-index: 10;
            }

            .fab:hover {
                background: var(--primary-dark);
                transform: scale(1.1);
            }

            @media (max-width: 768px) {
                .container {
                    padding: 16px;
                }

                .upload-zone {
                    padding: 32px 16px;
                }

                .fab {
                    bottom: 16px;
                    right: 16px;
                }
            }
        </style>
    </head>
    <body>
        <div class="app-bar">
            <h1>
                <span class="material-icons">camera_alt</span>
                PlateVision Pro
            </h1>
        </div>

        <div class="container">
            <div class="upload-card">
                <div class="card-header">
                    <div class="card-title">Image Upload</div>
                    <div class="card-subtitle">Select or drag an image to analyze license plates</div>
                </div>
                
                <div class="upload-zone" id="uploadZone">
                    <span class="material-icons upload-icon">cloud_upload</span>
                    <div class="upload-text">Drop image here or click to browse</div>
                    <div class="upload-hint">PNG, JPG, WEBP up to 10MB</div>
                    <input type="file" id="fileInput" accept="image/*">
                    <button class="btn" id="browseBtn">
                        <span class="material-icons">folder_open</span>
                        Browse Files
                    </button>
                </div>
            </div>

            <div class="preview-card" id="previewCard">
                <img id="previewImg" class="preview-img" alt="Preview">
                <div style="display: flex; gap: 12px; justify-content: center;">
                    <button class="btn" id="analyzeBtn">
                        <span class="material-icons">search</span>
                        Analyze
                    </button>
                    <button class="btn btn-secondary" id="clearBtn">
                        <span class="material-icons">clear</span>
                        Clear
                    </button>
                </div>
            </div>

            <div class="loading-card" id="loadingCard">
                <div class="progress-circle"></div>
                <div>Processing image...</div>
                <div style="color: var(--on-surface-variant); font-size: 14px; margin-top: 8px;">
                    AI is analyzing the image for license plates
                </div>
            </div>

            <div class="error-card" id="errorCard"></div>

            <div class="results-card" id="resultsCard">
                <img id="resultImage" class="result-image" alt="Analysis result">
                <div class="results-content">
                    <div class="results-header">
                        <span class="material-icons">check_circle</span>
                        <span class="results-title">Analysis Complete</span>
                        <span class="chip" id="plateCount">0 plates</span>
                    </div>
                    <div class="plate-list" id="plateList"></div>
                </div>
            </div>
        </div>

        <button class="fab" id="newAnalysis" style="display: none;" title="New Analysis">
            <span class="material-icons">add</span>
        </button>

        <script>
            const uploadZone = document.getElementById('uploadZone');
            const fileInput = document.getElementById('fileInput');
            const browseBtn = document.getElementById('browseBtn');
            const previewCard = document.getElementById('previewCard');
            const previewImg = document.getElementById('previewImg');
            const analyzeBtn = document.getElementById('analyzeBtn');
            const clearBtn = document.getElementById('clearBtn');
            const loadingCard = document.getElementById('loadingCard');
            const errorCard = document.getElementById('errorCard');
            const resultsCard = document.getElementById('resultsCard');
            const resultImage = document.getElementById('resultImage');
            const plateList = document.getElementById('plateList');
            const plateCount = document.getElementById('plateCount');
            const newAnalysis = document.getElementById('newAnalysis');

            let selectedFile = null;

            // Event listeners
            uploadZone.addEventListener('click', () => fileInput.click());
            browseBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                fileInput.click();
            });

            uploadZone.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadZone.classList.add('dragover');
            });

            uploadZone.addEventListener('dragleave', (e) => {
                if (!uploadZone.contains(e.relatedTarget)) {
                    uploadZone.classList.remove('dragover');
                }
            });

            uploadZone.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadZone.classList.remove('dragover');
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    handleFileSelect(files[0]);
                }
            });

            fileInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    handleFileSelect(e.target.files[0]);
                }
            });

            analyzeBtn.addEventListener('click', () => {
                if (selectedFile) {
                    analyzeImage(selectedFile);
                }
            });

            clearBtn.addEventListener('click', () => {
                resetInterface();
            });

            newAnalysis.addEventListener('click', () => {
                resetInterface();
            });

            function handleFileSelect(file) {
                if (!file.type.startsWith('image/')) {
                    showError('Please select a valid image file (PNG, JPG, WEBP)');
                    return;
                }

                if (file.size > 10 * 1024 * 1024) {
                    showError('File size must be less than 10MB');
                    return;
                }

                selectedFile = file;
                
                const reader = new FileReader();
                reader.onload = (e) => {
                    previewImg.src = e.target.result;
                    previewCard.style.display = 'block';
                    hideError();
                    hideResults();
                };
                reader.readAsDataURL(file);
            }

            async function analyzeImage(file) {
                hideError();
                hideResults();
                showLoading();

                const formData = new FormData();
                formData.append('file', file);

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });

                    const data = await response.json();

                    if (!response.ok) {
                        throw new Error(data.error || 'Analysis failed');
                    }

                    displayResults(data);
                } catch (err) {
                    showError(err.message);
                } finally {
                    hideLoading();
                }
            }

            function displayResults(data) {
                resultImage.src = data.annotated_image_url;
                plateList.innerHTML = '';
                
                if (data.results && data.results.length > 0) {
                    plateCount.textContent = `${data.results.length} plate${data.results.length > 1 ? 's' : ''}`;
                    
                    data.results.forEach((plate) => {
                        const plateItem = document.createElement('div');
                        plateItem.className = 'plate-item';
                        
                        const plateText = plate.text || 'Unreadable';
                        const confidence = plate.confidence ? Math.round(plate.confidence * 100) : 0;
                        const ocrConf = plate.ocr_conf ? Math.round(plate.ocr_conf * 100) : 0;
                        const detConf = plate.det_conf ? Math.round(plate.det_conf * 100) : 0;
                        
                        plateItem.innerHTML = `
                            <div class="plate-text">${plateText}</div>
                            <div>
                                <div class="confidence-badge" style="margin-bottom: 4px;">Combined: ${confidence}%</div>
                                <div class="confidence-badge" style="background: #2196f3;">OCR: ${ocrConf}%</div>
                                <div class="confidence-badge" style="background: #ff9800; margin-left: 8px;">Detection: ${detConf}%</div>
                            </div>
                        `;
                        
                        plateList.appendChild(plateItem);
                    });
                } else {
                    plateCount.textContent = '0 plates';
                    plateList.innerHTML = `
                        <div class="empty-state">
                            <span class="material-icons" style="font-size: 48px; margin-bottom: 16px; display: block;">search_off</span>
                            <div>No license plates detected</div>
                            <div style="font-size: 14px; margin-top: 8px;">Try uploading a clearer image</div>
                        </div>
                    `;
                }

                showResults();
            }

            function showLoading() {
                previewCard.style.display = 'none';
                loadingCard.style.display = 'block';
                analyzeBtn.disabled = true;
            }

            function hideLoading() {
                loadingCard.style.display = 'none';
                analyzeBtn.disabled = false;
            }

            function showResults() {
                resultsCard.style.display = 'block';
                newAnalysis.style.display = 'flex';
            }

            function hideResults() {
                resultsCard.style.display = 'none';
                newAnalysis.style.display = 'none';
            }

            function showError(message) {
                errorCard.textContent = message;
                errorCard.style.display = 'block';
                hideLoading();
            }

            function hideError() {
                errorCard.style.display = 'none';
            }

            function resetInterface() {
                selectedFile = null;
                fileInput.value = '';
                previewCard.style.display = 'none';
                hideLoading();
                hideResults();
                hideError();
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Convert uploaded file to OpenCV image
        img_bytes = await file.read()
        npimg = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        if img is None:
            return JSONResponse({"error": "Invalid image file"}, status_code=400)

        # Run detection + OCR
        results = pr.read_plate(img)
        
        # Draw detections and add confidence to results
        for det in results:
            x1, y1, x2, y2 = det["bbox"]
            text = det["text"]
            conf = det.get("ocr_conf", 0)      # fetch OCR confidence
            det_conf = det.get("det_conf", 0)  # fetch YOLO confidence

            # optional: combine both confidences
            combined = conf * det_conf
            det["confidence"] = combined  # Add combined confidence to the dict
            det["ocr_conf"] = conf  # Ensure keys are consistent
            det["det_conf"] = det_conf

            # show plate text with confidence percentage
            label = f"{text} ({combined*100:.1f}%)"

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


        # Save annotated image to static/ folder
        filename = f"{uuid.uuid4().hex}.jpg"
        save_path = os.path.join("static", filename)
        cv2.imwrite(save_path, img)

        return JSONResponse({
            "results": results,
            "annotated_image_url": f"/static/{filename}"
        })
    
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
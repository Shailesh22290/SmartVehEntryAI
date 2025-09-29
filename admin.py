from fastapi import FastAPI, Request, Form, HTTPException, Cookie
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import hashlib
import secrets
import logging
from datetime import datetime, timedelta
from fastapi import Depends
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from database import SessionLocal, VehicleLog

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

admin = FastAPI(title="PlateVision Pro - Admin Panel")

# Add CORS middleware
admin.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple session store (in production, use Redis or database)
active_sessions = {}

# Admin credentials (CHANGE THESE!)
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD_HASH = hashlib.sha256("admin123".encode()).hexdigest()

def verify_password(password: str) -> bool:
    """Verify admin password."""
    return hashlib.sha256(password.encode()).hexdigest() == ADMIN_PASSWORD_HASH

def create_session() -> str:
    """Create a new session token."""
    token = secrets.token_urlsafe(32)
    active_sessions[token] = datetime.now()
    return token

def verify_session(session_token: str = Cookie(None)) -> bool:
    """Verify if session is valid."""
    if not session_token or session_token not in active_sessions:
        return False
    # Check if session is not older than 24 hours
    if datetime.now() - active_sessions[session_token] > timedelta(hours=24):
        del active_sessions[session_token]
        return False
    return True

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@admin.get("/", response_class=HTMLResponse)
async def redirect_to_login():
    """Redirect root to login."""
    return RedirectResponse(url="/login")


@admin.get("/login", response_class=HTMLResponse)
async def admin_login_page():
    """Admin login page."""
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Admin Login - PlateVision Pro</title>
      <script src="https://cdn.tailwindcss.com"></script>
      <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
    </head>
    <body class="bg-gradient-to-br from-blue-500 to-purple-600 min-h-screen flex items-center justify-center">
      <div class="bg-white rounded-2xl shadow-2xl p-8 w-full max-w-md">
        <div class="text-center mb-8">
          <span class="material-icons text-blue-600 text-6xl">admin_panel_settings</span>
          <h1 class="text-3xl font-bold text-gray-800 mt-4">Admin Login</h1>
          <p class="text-gray-500 mt-2">PlateVision Pro Control Panel</p>
        </div>

        <form id="loginForm" class="space-y-6">
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-2">Username</label>
            <input type="text" name="username" required
              class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              placeholder="Enter username">
          </div>

          <div>
            <label class="block text-sm font-medium text-gray-700 mb-2">Password</label>
            <input type="password" name="password" required
              class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              placeholder="Enter password">
          </div>

          <button type="submit"
            class="w-full bg-blue-600 text-white py-3 rounded-lg hover:bg-blue-700 font-semibold flex items-center justify-center gap-2">
            <span class="material-icons">login</span>
            Sign In
          </button>
        </form>

        <div id="error" class="hidden mt-4 p-3 bg-red-100 text-red-700 rounded-lg text-sm"></div>
      </div>

      <script>
        document.getElementById('loginForm').addEventListener('submit', async (e) => {
          e.preventDefault();
          const formData = new FormData(e.target);
          
          try {
            const res = await fetch('/login', {
              method: 'POST',
              body: formData
            });
            
            if (res.ok) {
              window.location.href = '/dashboard';
            } else {
              const data = await res.json();
              document.getElementById('error').textContent = data.detail || 'Invalid credentials';
              document.getElementById('error').classList.remove('hidden');
            }
          } catch (err) {
            document.getElementById('error').textContent = 'Login failed';
            document.getElementById('error').classList.remove('hidden');
          }
        });
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@admin.post("/login")
async def admin_login(username: str = Form(...), password: str = Form(...)):
    """Handle admin login."""
    if username == ADMIN_USERNAME and verify_password(password):
        session_token = create_session()
        response = JSONResponse({"message": "Login successful"})
        response.set_cookie(
            key="session_token",
            value=session_token,
            httponly=True,
            max_age=86400,  # 24 hours
            samesite="lax"
        )
        return response
    raise HTTPException(status_code=401, detail="Invalid credentials")


@admin.get("/logout")
async def admin_logout(session_token: str = Cookie(None)):
    """Handle admin logout."""
    if session_token and session_token in active_sessions:
        del active_sessions[session_token]
    response = RedirectResponse(url="/login")
    response.delete_cookie("session_token")
    return response


@admin.get("/dashboard", response_class=HTMLResponse)
async def admin_dashboard(session_token: str = Cookie(None)):
    """Admin dashboard."""
    if not verify_session(session_token):
        return RedirectResponse(url="/login")
    
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Admin Dashboard - PlateVision Pro</title>
      <script src="https://cdn.tailwindcss.com"></script>
      <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
      <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    </head>
    <body class="bg-gray-100">
      <!-- Navbar -->
      <nav class="bg-white shadow-lg sticky top-0 z-50">
        <div class="container mx-auto px-6 py-4 flex justify-between items-center">
          <div class="flex items-center gap-3">
            <span class="material-icons text-blue-600 text-3xl">admin_panel_settings</span>
            <h1 class="text-2xl font-bold text-gray-800">Admin Dashboard</h1>
          </div>
          <div class="flex items-center gap-4">
            <span class="text-gray-600 text-sm">Admin Panel</span>
            <button onclick="logout()" class="bg-red-600 text-white px-4 py-2 rounded-lg hover:bg-red-700 flex items-center gap-2">
              <span class="material-icons text-sm">logout</span>
              Logout
            </button>
          </div>
        </div>
      </nav>

      <div class="container mx-auto px-6 py-8">
        <!-- Stats Cards -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div class="bg-white rounded-xl shadow p-6">
            <div class="flex items-center justify-between">
              <div>
                <p class="text-gray-500 text-sm">Total Vehicles</p>
                <p id="totalVehicles" class="text-3xl font-bold text-gray-800">0</p>
              </div>
              <span class="material-icons text-blue-600 text-5xl">directions_car</span>
            </div>
          </div>

          <div class="bg-white rounded-xl shadow p-6">
            <div class="flex items-center justify-between">
              <div>
                <p class="text-gray-500 text-sm">Currently Inside</p>
                <p id="insideVehicles" class="text-3xl font-bold text-green-600">0</p>
              </div>
              <span class="material-icons text-green-600 text-5xl">login</span>
            </div>
          </div>

          <div class="bg-white rounded-xl shadow p-6">
            <div class="flex items-center justify-between">
              <div>
                <p class="text-gray-500 text-sm">Exited Today</p>
                <p id="exitedVehicles" class="text-3xl font-bold text-orange-600">0</p>
              </div>
              <span class="material-icons text-orange-600 text-5xl">logout</span>
            </div>
          </div>

          <div class="bg-white rounded-xl shadow p-6">
            <div class="flex items-center justify-between">
              <div>
                <p class="text-gray-500 text-sm">Today's Total</p>
                <p id="todayTotal" class="text-3xl font-bold text-purple-600">0</p>
              </div>
              <span class="material-icons text-purple-600 text-5xl">today</span>
            </div>
          </div>
        </div>

        <!-- Charts -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          <div class="bg-white rounded-xl shadow p-6">
            <h3 class="text-lg font-semibold mb-4">Vehicle Types</h3>
            <canvas id="vehicleTypeChart"></canvas>
          </div>
          <div class="bg-white rounded-xl shadow p-6">
            <h3 class="text-lg font-semibold mb-4">Entry/Exit Status</h3>
            <canvas id="statusChart"></canvas>
          </div>
        </div>

        <!-- Filters & Actions -->
        <div class="bg-white rounded-xl shadow p-6 mb-6">
          <div class="flex flex-wrap gap-4 items-end">
            <div class="flex-1 min-w-[200px]">
              <label class="block text-sm font-medium mb-2">Filter by Status</label>
              <select id="statusFilter" class="w-full border rounded-lg p-2">
                <option value="">All</option>
                <option value="ENTRY">Currently Inside (ENTRY)</option>
                <option value="EXIT">Exited (EXIT)</option>
              </select>
            </div>
            <div class="flex-1 min-w-[200px]">
              <label class="block text-sm font-medium mb-2">Filter by Date</label>
              <input type="date" id="dateFilter" class="w-full border rounded-lg p-2">
            </div>
            <div class="flex-1 min-w-[200px]">
              <label class="block text-sm font-medium mb-2">Search Vehicle</label>
              <input type="text" id="searchFilter" placeholder="Vehicle number..." class="w-full border rounded-lg p-2">
            </div>
            <button onclick="applyFilters()" class="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 flex items-center gap-2">
              <span class="material-icons">filter_list</span>
              Apply
            </button>
            <button onclick="exportReport()" class="bg-green-600 text-white px-6 py-2 rounded-lg hover:bg-green-700 flex items-center gap-2">
              <span class="material-icons">download</span>
              Export CSV
            </button>
          </div>
        </div>

        <!-- Vehicle Table -->
        <div class="bg-white rounded-xl shadow overflow-hidden">
          <div class="p-6 border-b">
            <h3 class="text-lg font-semibold">Vehicle Logs</h3>
          </div>
          <div class="overflow-x-auto">
            <table class="w-full">
              <thead class="bg-gray-50">
                <tr>
                  <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">ID</th>
                  <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Vehicle No.</th>
                  <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Driver</th>
                  <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Type</th>
                  <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Entry Time</th>
                  <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Exit Time</th>
                  <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
                  <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Actions</th>
                </tr>
              </thead>
              <tbody id="vehicleTableBody" class="divide-y divide-gray-200">
                <!-- Data will be populated here -->
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <!-- Edit Modal -->
      <div id="editModal" class="hidden fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div class="bg-white rounded-xl p-6 w-full max-w-md">
          <h3 class="text-xl font-semibold mb-4">Edit Vehicle Log</h3>
          <form id="editForm" class="space-y-4">
            <input type="hidden" id="editId">
            <div>
              <label class="block text-sm font-medium mb-2">Vehicle Number</label>
              <input type="text" id="editVehicleNumber" class="w-full border rounded-lg p-2" readonly>
            </div>
            <div>
              <label class="block text-sm font-medium mb-2">Driver Name</label>
              <input type="text" id="editDriverName" class="w-full border rounded-lg p-2">
            </div>
            <div>
              <label class="block text-sm font-medium mb-2">Vehicle Type</label>
              <select id="editVehicleType" class="w-full border rounded-lg p-2">
                <option>Car</option>
                <option>Bus</option>
                <option>Truck</option>
                <option>Auto</option>
                <option>Two-Wheeler</option>
                <option>Other</option>
              </select>
            </div>
            <div>
              <label class="block text-sm font-medium mb-2">Remarks</label>
              <textarea id="editRemarks" rows="2" class="w-full border rounded-lg p-2"></textarea>
            </div>
            <div class="flex gap-4">
              <button type="submit" class="flex-1 bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700">Save</button>
              <button type="button" onclick="closeEditModal()" class="flex-1 bg-gray-200 py-2 rounded-lg hover:bg-gray-300">Cancel</button>
            </div>
          </form>
        </div>
      </div>

      <script>
        let allVehicles = [];
        let vehicleTypeChart, statusChart;

        async function loadData() {
          try {
            const res = await fetch('/api/vehicles');
            allVehicles = await res.json();
            updateStats();
            updateCharts();
            displayVehicles(allVehicles);
          } catch (err) {
            console.error('Failed to load data:', err);
          }
        }

        function updateStats() {
          const total = allVehicles.length;
          const inside = allVehicles.filter(v => v.status === 'ENTRY').length;
          const exited = allVehicles.filter(v => v.status === 'EXIT').length;
          
          const today = new Date().toDateString();
          const todayVehicles = allVehicles.filter(v => 
            new Date(v.entry_time).toDateString() === today
          ).length;

          document.getElementById('totalVehicles').textContent = total;
          document.getElementById('insideVehicles').textContent = inside;
          document.getElementById('exitedVehicles').textContent = exited;
          document.getElementById('todayTotal').textContent = todayVehicles;
        }

        function updateCharts() {
          // Vehicle Type Chart
          const types = {};
          allVehicles.forEach(v => {
            if (v.vehicle_type) {
              types[v.vehicle_type] = (types[v.vehicle_type] || 0) + 1;
            }
          });

          if (vehicleTypeChart) vehicleTypeChart.destroy();
          vehicleTypeChart = new Chart(document.getElementById('vehicleTypeChart'), {
            type: 'doughnut',
            data: {
              labels: Object.keys(types),
              datasets: [{
                data: Object.values(types),
                backgroundColor: ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#6B7280']
              }]
            },
            options: {
              responsive: true,
              maintainAspectRatio: true
            }
          });

          // Status Chart
          const statusData = {
            ENTRY: allVehicles.filter(v => v.status === 'ENTRY').length,
            EXIT: allVehicles.filter(v => v.status === 'EXIT').length
          };

          if (statusChart) statusChart.destroy();
          statusChart = new Chart(document.getElementById('statusChart'), {
            type: 'bar',
            data: {
              labels: ['Currently Inside', 'Exited'],
              datasets: [{
                label: 'Vehicles',
                data: [statusData.ENTRY, statusData.EXIT],
                backgroundColor: ['#10B981', '#F59E0B']
              }]
            },
            options: {
              responsive: true,
              maintainAspectRatio: true,
              scales: {
                y: { beginAtZero: true }
              }
            }
          });
        }

        function displayVehicles(vehicles) {
          const tbody = document.getElementById('vehicleTableBody');
          tbody.innerHTML = vehicles.map(v => `
            <tr class="hover:bg-gray-50">
              <td class="px-6 py-4 text-sm">${v.id}</td>
              <td class="px-6 py-4 text-sm font-mono font-semibold">${v.vehicle_number}</td>
              <td class="px-6 py-4 text-sm">${v.driver_name || '-'}</td>
              <td class="px-6 py-4 text-sm">${v.vehicle_type || '-'}</td>
              <td class="px-6 py-4 text-sm">${v.entry_time ? new Date(v.entry_time).toLocaleString() : '-'}</td>
              <td class="px-6 py-4 text-sm">${v.exit_time ? new Date(v.exit_time).toLocaleString() : '-'}</td>
              <td class="px-6 py-4">
                <span class="px-2 py-1 rounded-full text-xs font-semibold ${v.status === 'ENTRY' ? 'bg-green-100 text-green-800' : 'bg-orange-100 text-orange-800'}">
                  ${v.status}
                </span>
              </td>
              <td class="px-6 py-4">
                <button onclick="editVehicle(${v.id})" class="text-blue-600 hover:underline mr-2">Edit</button>
                <button onclick="deleteVehicle(${v.id})" class="text-red-600 hover:underline">Delete</button>
              </td>
            </tr>
          `).join('');
        }

        function applyFilters() {
          let filtered = [...allVehicles];
          
          const status = document.getElementById('statusFilter').value;
          if (status) {
            filtered = filtered.filter(v => v.status === status);
          }

          const date = document.getElementById('dateFilter').value;
          if (date) {
            filtered = filtered.filter(v => 
              new Date(v.entry_time).toDateString() === new Date(date).toDateString()
            );
          }

          const search = document.getElementById('searchFilter').value.toUpperCase();
          if (search) {
            filtered = filtered.filter(v => v.vehicle_number.includes(search));
          }

          displayVehicles(filtered);
        }

        async function editVehicle(id) {
          const vehicle = allVehicles.find(v => v.id === id);
          if (!vehicle) return;

          document.getElementById('editId').value = vehicle.id;
          document.getElementById('editVehicleNumber').value = vehicle.vehicle_number;
          document.getElementById('editDriverName').value = vehicle.driver_name || '';
          document.getElementById('editVehicleType').value = vehicle.vehicle_type || '';
          document.getElementById('editRemarks').value = vehicle.remarks || '';
          document.getElementById('editModal').classList.remove('hidden');
        }

        function closeEditModal() {
          document.getElementById('editModal').classList.add('hidden');
        }

        document.getElementById('editForm').addEventListener('submit', async (e) => {
          e.preventDefault();
          const id = document.getElementById('editId').value;
          const data = {
            driver_name: document.getElementById('editDriverName').value,
            vehicle_type: document.getElementById('editVehicleType').value,
            remarks: document.getElementById('editRemarks').value
          };

          try {
            const res = await fetch(`/api/vehicles/${id}`, {
              method: 'PUT',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(data)
            });

            if (res.ok) {
              alert('✅ Updated successfully!');
              closeEditModal();
              loadData();
            }
          } catch (err) {
            alert('❌ Update failed: ' + err.message);
          }
        });

        async function deleteVehicle(id) {
          if (!confirm('Are you sure you want to delete this record?')) return;

          try {
            const res = await fetch(`/api/vehicles/${id}`, { method: 'DELETE' });
            if (res.ok) {
              alert('✅ Deleted successfully!');
              loadData();
            }
          } catch (err) {
            alert('❌ Delete failed: ' + err.message);
          }
        }

        function exportReport() {
          const csv = [
            ['ID', 'Vehicle Number', 'Driver', 'Type', 'Entry Time', 'Exit Time', 'Status', 'Remarks'],
            ...allVehicles.map(v => [
              v.id,
              v.vehicle_number,
              v.driver_name || '',
              v.vehicle_type || '',
              v.entry_time || '',
              v.exit_time || '',
              v.status,
              v.remarks || ''
            ])
          ].map(row => row.join(',')).join('\\n');

          const blob = new Blob([csv], { type: 'text/csv' });
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = `vehicle_report_${new Date().toISOString().split('T')[0]}.csv`;
          a.click();
        }

        async function logout() {
          await fetch('/logout');
          window.location.href = '/login';
        }

        // Load data on page load
        loadData();
        setInterval(loadData, 30000); // Refresh every 30 seconds
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@admin.get("/api/vehicles")
async def get_all_vehicles(session_token: str = Cookie(None), db: Session = Depends(get_db)):
    """Get all vehicle logs for admin."""
    if not verify_session(session_token):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        vehicles = db.query(VehicleLog).order_by(VehicleLog.entry_time.desc()).all()
        return [{
            "id": v.id,
            "vehicle_number": v.vehicle_number,
            "driver_name": v.driver_name,
            "vehicle_type": v.vehicle_type,
            "entry_time": v.entry_time.isoformat() if v.entry_time else None,
            "exit_time": v.exit_time.isoformat() if v.exit_time else None,
            "status": v.status,
            "remarks": v.remarks,
            "operator_id": v.operator_id,
            "gate_id": v.gate_id
        } for v in vehicles]
    except Exception as e:
        logger.error(f"Error fetching vehicles: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch vehicles")


@admin.put("/api/vehicles/{vehicle_id}")
async def update_vehicle(
    vehicle_id: int,
    data: dict,
    session_token: str = Cookie(None),
    db: Session = Depends(get_db)
):
    """Update vehicle log."""
    if not verify_session(session_token):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        vehicle = db.query(VehicleLog).filter(VehicleLog.id == vehicle_id).first()
        if not vehicle:
            raise HTTPException(status_code=404, detail="Vehicle not found")
        
        if "driver_name" in data:
            vehicle.driver_name = data["driver_name"]
        if "vehicle_type" in data:
            vehicle.vehicle_type = data["vehicle_type"]
        if "remarks" in data:
            vehicle.remarks = data["remarks"]
        
        db.commit()
        return {"message": "Updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating vehicle: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Update failed")


@admin.delete("/api/vehicles/{vehicle_id}")
async def delete_vehicle(
    vehicle_id: int,
    session_token: str = Cookie(None),
    db: Session = Depends(get_db)
):
    """Delete vehicle log."""
    if not verify_session(session_token):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        vehicle = db.query(VehicleLog).filter(VehicleLog.id == vehicle_id).first()
        if not vehicle:
            raise HTTPException(status_code=404, detail="Vehicle not found")
        
        db.delete(vehicle)
        db.commit()
        return {"message": "Deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting vehicle: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Delete failed")


@admin.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "admin", "timestamp": datetime.now().isoformat()}
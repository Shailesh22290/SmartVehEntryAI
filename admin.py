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
from sqlalchemy import or_, and_
from database import SessionLocal, VehicleLog, BannedVehicle

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

# Admin credentials (CHANGE THESE IN PRODUCTION!)
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
    return RedirectResponse(url="/admin/login")

@admin.get("/login", response_class=HTMLResponse)
async def admin_login_page(request: Request, username: str = None, password: str = None):
    """Admin login page or handle GET login for testing."""
    if username and password:
        if username == ADMIN_USERNAME and verify_password(password):
            session_token = create_session()
            response = RedirectResponse(url="/admin/dashboard")
            response.set_cookie(
                key="session_token",
                value=session_token,
                httponly=True,
                max_age=86400,
                samesite="lax",
                secure=False
            )
            return response
        else:
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
                <div id="error" class="mt-4 p-3 bg-red-100 text-red-700 rounded-lg text-sm">Invalid credentials</div>
              </div>
              <script>
                document.getElementById('loginForm').addEventListener('submit', async (e) => {
                  e.preventDefault();
                  const formData = new FormData(e.target);
                  try {
                    const res = await fetch('/admin/login', {
                      method: 'POST',
                      body: formData
                    });
                    if (res.ok) {
                      window.location.href = '/admin/dashboard';
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
            const res = await fetch('/admin/login', {
              method: 'POST',
              body: formData
            });
            if (res.ok) {
              window.location.href = '/admin/dashboard';
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
            max_age=86400,
            samesite="lax",
            secure=False
        )
        return response
    raise HTTPException(status_code=401, detail="Invalid credentials")

@admin.get("/logout")
async def admin_logout(session_token: str = Cookie(None)):
    """Handle admin logout."""
    if session_token and session_token in active_sessions:
        del active_sessions[session_token]
    response = RedirectResponse(url="/admin/login")
    response.delete_cookie("session_token")
    return response

@admin.get("/dashboard", response_class=HTMLResponse)
async def admin_dashboard(session_token: str = Cookie(None), db: Session = Depends(get_db)):
    """Admin dashboard."""
    if not verify_session(session_token):
        return RedirectResponse(url="/admin/login")
    
    try:
        total_vehicles = db.query(VehicleLog).count()
        inside_vehicles = db.query(VehicleLog).filter(
            VehicleLog.status == "ENTRY", 
            VehicleLog.exit_time == None
        ).count()
        today = datetime.now().date()
        exited_today = db.query(VehicleLog).filter(
            VehicleLog.status == "EXIT",
            VehicleLog.exit_time >= today
        ).count()
        today_total = db.query(VehicleLog).filter(
            VehicleLog.entry_time >= today
        ).count()

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Admin Dashboard - PlateVision Pro</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body class="bg-gray-100">
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
    <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
      <div class="bg-white rounded-xl shadow p-6">
        <div class="flex items-center justify-between">
          <div>
            <p class="text-gray-500 text-sm">Total Vehicles</p>
            <p id="totalVehicles" class="text-3xl font-bold text-gray-800">{total_vehicles}</p>
          </div>
          <span class="material-icons text-blue-600 text-5xl">directions_car</span>
        </div>
      </div>
      <div class="bg-white rounded-xl shadow p-6">
        <div class="flex items-center justify-between">
          <div>
            <p class="text-gray-500 text-sm">Currently Inside</p>
            <p id="insideVehicles" class="text-3xl font-bold text-green-600">{inside_vehicles}</p>
          </div>
          <span class="material-icons text-green-600 text-5xl">login</span>
        </div>
      </div>
      <div class="bg-white rounded-xl shadow p-6">
        <div class="flex items-center justify-between">
          <div>
            <p class="text-gray-500 text-sm">Exited Today</p>
            <p id="exitedVehicles" class="text-3xl font-bold text-orange-600">{exited_today}</p>
          </div>
          <span class="material-icons text-orange-600 text-5xl">logout</span>
        </div>
      </div>
      <div class="bg-white rounded-xl shadow p-6">
        <div class="flex items-center justify-between">
          <div>
            <p class="text-gray-500 text-sm">Today's Total</p>
            <p id="todayTotal" class="text-3xl font-bold text-purple-600">{today_total}</p>
          </div>
          <span class="material-icons text-purple-600 text-5xl">today</span>
        </div>
      </div>
    </div>

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
              <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Ban Status</th>
              <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Actions</th>
            </tr>
          </thead>
          <tbody id="vehicleTableBody" class="divide-y divide-gray-200">
          </tbody>
        </table>
      </div>
    </div>

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

    <div id="banModal" class="hidden fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div class="bg-white rounded-xl p-6 w-full max-w-md">
        <h3 class="text-xl font-semibold mb-4">Ban Vehicle</h3>
        <form id="banForm" class="space-y-4">
          <input type="hidden" id="banVehicleNumber">
          <div>
            <label class="block text-sm font-medium mb-2">Vehicle Number</label>
            <input type="text" id="banVehicleNumberDisplay" class="w-full border rounded-lg p-2" readonly>
          </div>
          <div>
            <label class="block text-sm font-medium mb-2">Reason for Ban</label>
            <textarea id="banReason" rows="3" class="w-full border rounded-lg p-2" placeholder="Enter ban reason..." required></textarea>
          </div>
          <div class="flex gap-4">
            <button type="submit" class="flex-1 bg-red-600 text-white py-2 rounded-lg hover:bg-red-700">Ban Vehicle</button>
            <button type="button" onclick="closeBanModal()" class="flex-1 bg-gray-200 py-2 rounded-lg hover:bg-gray-300">Cancel</button>
          </div>
        </form>
      </div>
    </div>

    <div id="historyModal" class="hidden fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div class="bg-white rounded-xl p-6 w-full max-w-3xl max-h-[80vh] overflow-y-auto">
        <h3 class="text-xl font-semibold mb-4">Vehicle Travel History</h3>
        <div class="overflow-x-auto">
          <table class="w-full">
            <thead class="bg-gray-50">
              <tr>
                <th class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">ID</th>
                <th class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Entry Time</th>
                <th class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Exit Time</th>
                <th class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Driver</th>
                <th class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Type</th>
                <th class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
                <th class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Remarks</th>
              </tr>
            </thead>
            <tbody id="historyTableBody" class="divide-y divide-gray-200">
            </tbody>
          </table>
        </div>
        <div class="mt-4 flex justify-end">
          <button onclick="closeHistoryModal()" class="bg-gray-200 py-2 px-6 rounded-lg hover:bg-gray-300">Close</button>
        </div>
      </div>
    </div>

    <div id="errorToast" class="hidden fixed bottom-4 right-4 bg-red-600 text-white rounded-lg p-4 flex items-center gap-2 shadow-lg z-50">
      <span class="material-icons">error</span>
      <span id="errorMessage"></span>
    </div>

    <div id="successToast" class="hidden fixed bottom-4 right-4 bg-green-600 text-white rounded-lg p-4 flex items-center gap-2 shadow-lg z-50">
      <span class="material-icons">check_circle</span>
      <span id="successMessage"></span>
    </div>
  </div>

  <script>
    let allVehicles = [];
    let vehicleTypeChart, statusChart;

    async function loadData() {{
      try {{
        const res = await fetch('/admin/api/vehicles');
        if (!res.ok) throw new Error('Failed to fetch vehicles');
        allVehicles = await res.json();
        updateStats();
        updateCharts();
        displayVehicles(allVehicles);
      }} catch (err) {{
        showError('Failed to load data: ' + err.message);
        console.error('Load data error:', err);
      }}
    }}

    function updateStats() {{
      const total = allVehicles.length;
      const inside = allVehicles.filter(v => v.status === 'ENTRY' && !v.exit_time).length;
      const today = new Date().toDateString();
      const exited = allVehicles.filter(v => 
        v.status === 'EXIT' && v.exit_time && new Date(v.exit_time).toDateString() === today
      ).length;
      const todayVehicles = allVehicles.filter(v => 
        new Date(v.entry_time).toDateString() === today
      ).length;

      document.getElementById('totalVehicles').textContent = total;
      document.getElementById('insideVehicles').textContent = inside;
      document.getElementById('exitedVehicles').textContent = exited;
      document.getElementById('todayTotal').textContent = todayVehicles;
    }}

    function updateCharts() {{
      const types = {{}};
      allVehicles.forEach(v => {{
        if (v.vehicle_type) {{
          types[v.vehicle_type] = (types[v.vehicle_type] || 0) + 1;
        }}
      }});

      if (vehicleTypeChart) vehicleTypeChart.destroy();
      const typeCtx = document.getElementById('vehicleTypeChart');
      if (typeCtx && Object.keys(types).length > 0) {{
        vehicleTypeChart = new Chart(typeCtx, {{
          type: 'doughnut',
          data: {{
            labels: Object.keys(types),
            datasets: [{{
              data: Object.values(types),
              backgroundColor: ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#6B7280']
            }}]
          }},
          options: {{
            responsive: true,
            maintainAspectRatio: true,
            plugins: {{
              legend: {{
                position: 'bottom'
              }}
            }}
          }}
        }});
      }}

      const statusData = {{
        ENTRY: allVehicles.filter(v => v.status === 'ENTRY' && !v.exit_time).length,
        EXIT: allVehicles.filter(v => v.status === 'EXIT' || v.exit_time).length
      }};

      if (statusChart) statusChart.destroy();
      const statusCtx = document.getElementById('statusChart');
      if (statusCtx) {{
        statusChart = new Chart(statusCtx, {{
          type: 'bar',
          data: {{
            labels: ['Currently Inside', 'Exited'],
            datasets: [{{
              label: 'Vehicles',
              data: [statusData.ENTRY, statusData.EXIT],
              backgroundColor: ['#10B981', '#F59E0B']
            }}]
          }},
          options: {{
            responsive: true,
            maintainAspectRatio: true,
            scales: {{
              y: {{ 
                beginAtZero: true,
                ticks: {{
                  stepSize: 1
                }}
              }}
            }}
          }}
        }});
      }}
    }}

    async function displayVehicles(vehicles) {{
      try {{
        const bannedRes = await fetch('/admin/api/banned-vehicles');
        const bannedVehicles = bannedRes.ok ? await bannedRes.json() : [];
        
        const tbody = document.getElementById('vehicleTableBody');
        if (vehicles.length === 0) {{
          tbody.innerHTML = '<tr><td colspan="9" class="px-6 py-8 text-center text-gray-500">No vehicle records found</td></tr>';
          return;
        }}
        
        tbody.innerHTML = vehicles.map(v => {{
          const isBanned = bannedVehicles.some(b => b.vehicle_number === v.vehicle_number);
          return `
            <tr class="hover:bg-gray-50">
              <td class="px-6 py-4 text-sm">${{v.id}}</td>
              <td class="px-6 py-4 text-sm font-mono font-semibold">${{v.vehicle_number || 'N/A'}}</td>
              <td class="px-6 py-4 text-sm">${{v.driver_name || '-'}}</td>
              <td class="px-6 py-4 text-sm">${{v.vehicle_type || '-'}}</td>
              <td class="px-6 py-4 text-sm">${{v.entry_time ? new Date(v.entry_time).toLocaleString() : '-'}}</td>
              <td class="px-6 py-4 text-sm">${{v.exit_time ? new Date(v.exit_time).toLocaleString() : '-'}}</td>
              <td class="px-6 py-4">
                <span class="px-2 py-1 rounded-full text-xs font-semibold ${{v.exit_time ? 'bg-orange-100 text-orange-800' : 'bg-green-100 text-green-800'}}">
                  ${{v.exit_time ? 'EXIT' : 'ENTRY'}}
                </span>
              </td>
              <td class="px-6 py-4">
                <span class="px-2 py-1 rounded-full text-xs font-semibold ${{isBanned ? 'bg-red-100 text-red-800' : 'bg-gray-100 text-gray-800'}}">
                  ${{isBanned ? 'BANNED' : 'ALLOWED'}}
                </span>
              </td>
              <td class="px-6 py-4">
                <div class="flex flex-wrap gap-2">
                  <button onclick="showHistory('${{v.vehicle_number}}')" class="text-purple-600 hover:underline text-sm">History</button>
                  <button onclick="editVehicle(${{v.id}})" class="text-blue-600 hover:underline text-sm">Edit</button>
                  <button onclick="deleteVehicle(${{v.id}})" class="text-red-600 hover:underline text-sm">Delete</button>
                  <button onclick="${{isBanned ? `unbanVehicle('${{v.vehicle_number}}')` : `banVehicle('${{v.vehicle_number}}')`}}" 
                          class="text-${{isBanned ? 'green' : 'red'}}-600 hover:underline text-sm">
                    ${{isBanned ? 'Unban' : 'Ban'}}
                  </button>
                </div>
              </td>
            </tr>
          `;
        }}).join('');
      }} catch (err) {{
        showError('Failed to display vehicles: ' + err.message);
        console.error('Display vehicles error:', err);
      }}
    }}

    function applyFilters() {{
      let filtered = [...allVehicles];
      
      const status = document.getElementById('statusFilter').value;
      if (status === 'ENTRY') {{
        filtered = filtered.filter(v => !v.exit_time);
      }} else if (status === 'EXIT') {{
        filtered = filtered.filter(v => v.exit_time);
      }}

      const date = document.getElementById('dateFilter').value;
      if (date) {{
        filtered = filtered.filter(v => 
          new Date(v.entry_time).toDateString() === new Date(date).toDateString()
        );
      }}

      const search = document.getElementById('searchFilter').value.toUpperCase();
      if (search) {{
        filtered = filtered.filter(v => v.vehicle_number && v.vehicle_number.includes(search));
      }}

      displayVehicles(filtered);
    }}

    async function showHistory(vehicleNumber) {{
      try {{
        const res = await fetch(`/admin/api/vehicle-history/${{encodeURIComponent(vehicleNumber)}}`);
        if (!res.ok) throw new Error('Failed to fetch history');
        const history = await res.json();
        
        const tbody = document.getElementById('historyTableBody');
        if (history.length === 0) {{
          tbody.innerHTML = '<tr><td colspan="7" class="px-4 py-4 text-center text-gray-500">No history found</td></tr>';
        }} else {{
          tbody.innerHTML = history.map(h => `
            <tr>
              <td class="px-4 py-2 text-sm">${{h.id}}</td>
              <td class="px-4 py-2 text-sm">${{h.entry_time ? new Date(h.entry_time).toLocaleString() : '-'}}</td>
              <td class="px-4 py-2 text-sm">${{h.exit_time ? new Date(h.exit_time).toLocaleString() : '-'}}</td>
              <td class="px-4 py-2 text-sm">${{h.driver_name || '-'}}</td>
              <td class="px-4 py-2 text-sm">${{h.vehicle_type || '-'}}</td>
              <td class="px-4 py-2 text-sm">${{h.exit_time ? 'EXIT' : 'ENTRY'}}</td>
              <td class="px-4 py-2 text-sm">${{h.remarks || '-'}}</td>
            </tr>
          `).join('');
        }}
        document.getElementById('historyModal').classList.remove('hidden');
      }} catch (err) {{
        showError('Failed to load history: ' + err.message);
        console.error('History error:', err);
      }}
    }}

    function closeHistoryModal() {{
      document.getElementById('historyModal').classList.add('hidden');
    }}

    async function banVehicle(vehicleNumber) {{
      document.getElementById('banVehicleNumber').value = vehicleNumber;
      document.getElementById('banVehicleNumberDisplay').value = vehicleNumber;
      document.getElementById('banModal').classList.remove('hidden');
    }}

    async function unbanVehicle(vehicleNumber) {{
      if (!confirm(`Are you sure you want to unban vehicle ${{vehicleNumber}}?`)) return;
      try {{
        const res = await fetch(`/admin/api/banned-vehicles/${{encodeURIComponent(vehicleNumber)}}`, {{ method: 'DELETE' }});
        if (res.ok) {{
          showSuccess(`Vehicle ${{vehicleNumber}} unbanned successfully`);
          loadData();
        }} else {{
          const data = await res.json();
          showError(data.detail || 'Failed to unban vehicle');
        }}
      }} catch (err) {{
        showError('Failed to unban vehicle: ' + err.message);
        console.error('Unban error:', err);
      }}
    }}

    document.getElementById('banForm').addEventListener('submit', async (e) => {{
      e.preventDefault();
      const vehicleNumber = document.getElementById('banVehicleNumber').value;
      const reason = document.getElementById('banReason').value;
      
      try {{
        const res = await fetch('/admin/api/banned-vehicles', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ vehicle_number: vehicleNumber, reason }})
        }});
        
        if (res.ok) {{
          showSuccess(`Vehicle ${{vehicleNumber}} banned successfully`);
          document.getElementById('banModal').classList.add('hidden');
          document.getElementById('banForm').reset();
          loadData();
        }} else {{
          const data = await res.json();
          showError(data.detail || 'Failed to ban vehicle');
        }}
      }} catch (err) {{
        showError('Failed to ban vehicle: ' + err.message);
        console.error('Ban error:', err);
      }}
    }});

    function closeBanModal() {{
      document.getElementById('banModal').classList.add('hidden');
      document.getElementById('banForm').reset();
    }}

    async function editVehicle(id) {{
      const vehicle = allVehicles.find(v => v.id === id);
      if (!vehicle) {{
        showError('Vehicle not found');
        return;
      }}

      document.getElementById('editId').value = vehicle.id;
      document.getElementById('editVehicleNumber').value = vehicle.vehicle_number;
      document.getElementById('editDriverName').value = vehicle.driver_name || '';
      document.getElementById('editVehicleType').value = vehicle.vehicle_type || '';
      document.getElementById('editRemarks').value = vehicle.remarks || '';
      document.getElementById('editModal').classList.remove('hidden');
    }}

    function closeEditModal() {{
      document.getElementById('editModal').classList.add('hidden');
    }}

    document.getElementById('editForm').addEventListener('submit', async (e) => {{
      e.preventDefault();
      const id = document.getElementById('editId').value;
      const data = {{
        driver_name: document.getElementById('editDriverName').value,
        vehicle_type: document.getElementById('editVehicleType').value,
        remarks: document.getElementById('editRemarks').value
      }};

      try {{
        const res = await fetch(`/admin/api/vehicles/${{id}}`, {{
          method: 'PUT',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify(data)
        }});

        if (res.ok) {{
          showSuccess('Updated successfully');
          closeEditModal();
          loadData();
        }} else {{
          const resData = await res.json();
          showError(resData.detail || 'Update failed');
        }}
      }} catch (err) {{
        showError('Update failed: ' + err.message);
        console.error('Update error:', err);
      }}
    }});

    async function deleteVehicle(id) {{
      if (!confirm('Are you sure you want to delete this record?')) return;

      try {{
        const res = await fetch(`/admin/api/vehicles/${{id}}`, {{ method: 'DELETE' }});
        if (res.ok) {{
          showSuccess('Deleted successfully');
          loadData();
        }} else {{
          const data = await res.json();
          showError(data.detail || 'Delete failed');
        }}
      }} catch (err) {{
        showError('Delete failed: ' + err.message);
        console.error('Delete error:', err);
      }}
    }}

    function exportReport() {{
      const csv = [
        ['ID', 'Vehicle Number', 'Driver', 'Type', 'Entry Time', 'Exit Time', 'Status', 'Remarks'],
        ...allVehicles.map(v => [
          v.id,
          v.vehicle_number || '',
          v.driver_name || '',
          v.vehicle_type || '',
          v.entry_time || '',
          v.exit_time || '',
          v.exit_time ? 'EXIT' : 'ENTRY',
          v.remarks || ''
        ])
      ].map(row => row.join(',')).join('\\n');

      const blob = new Blob([csv], {{ type: 'text/csv' }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `vehicle_report_${{new Date().toISOString().split('T')[0]}}.csv`;
      a.click();
      URL.revokeObjectURL(url);
    }}

    async function logout() {{
      await fetch('/admin/logout');
      window.location.href = '/admin/login';
    }}

    function showError(msg) {{
      const toast = document.getElementById('errorToast');
      document.getElementById('errorMessage').textContent = msg;
      toast.classList.remove('hidden');
      setTimeout(() => toast.classList.add('hidden'), 5000);
    }}

    function showSuccess(msg) {{
      const toast = document.getElementById('successToast');
      document.getElementById('successMessage').textContent = msg;
      toast.classList.remove('hidden');
      setTimeout(() => toast.classList.add('hidden'), 5000);
    }}

    loadData();
    setInterval(loadData, 30000);
  </script>
</body>
</html>"""
        return HTMLResponse(content=html)
    except Exception as e:
        logger.error(f"Error rendering dashboard: {e}")
        raise HTTPException(status_code=500, detail="Failed to render dashboard")

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

@admin.get("/api/vehicle-history/{vehicle_number}")
async def get_vehicle_history(vehicle_number: str, session_token: str = Cookie(None), db: Session = Depends(get_db)):
    """Get travel history for a specific vehicle."""
    if not verify_session(session_token):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        vehicles = db.query(VehicleLog).filter(
            VehicleLog.vehicle_number == vehicle_number
        ).order_by(VehicleLog.entry_time.desc()).all()
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
        logger.error(f"Error fetching vehicle history: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch vehicle history")

@admin.get("/api/banned-vehicles")
async def get_banned_vehicles(session_token: str = Cookie(None), db: Session = Depends(get_db)):
    """Get all banned vehicles."""
    if not verify_session(session_token):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        banned = db.query(BannedVehicle).all()
        return [{
            "id": b.id,
            "vehicle_number": b.vehicle_number,
            "reason": b.reason,
            "banned_at": b.banned_at.isoformat(),
            "banned_by": b.banned_by
        } for b in banned]
    except Exception as e:
        logger.error(f"Error fetching banned vehicles: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch banned vehicles")

@admin.post("/api/banned-vehicles")
async def ban_vehicle(
    data: dict,
    session_token: str = Cookie(None),
    db: Session = Depends(get_db)
):
    """Ban a vehicle."""
    if not verify_session(session_token):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        vehicle_number = data.get("vehicle_number", "").strip().upper()
        reason = data.get("reason", "").strip()
        
        if not vehicle_number:
            raise HTTPException(status_code=400, detail="Vehicle number is required")
        if not reason:
            raise HTTPException(status_code=400, detail="Ban reason is required")
        
        existing = db.query(BannedVehicle).filter(BannedVehicle.vehicle_number == vehicle_number).first()
        if existing:
            raise HTTPException(status_code=400, detail="Vehicle is already banned")
        
        banned_vehicle = BannedVehicle(
            vehicle_number=vehicle_number,
            reason=reason,
            banned_at=datetime.now(),
            banned_by="admin"
        )
        db.add(banned_vehicle)
        db.commit()
        return {"message": f"Vehicle {vehicle_number} banned successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error banning vehicle: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to ban vehicle")

@admin.delete("/api/banned-vehicles/{vehicle_number}")
async def unban_vehicle(
    vehicle_number: str,
    session_token: str = Cookie(None),
    db: Session = Depends(get_db)
):
    """Unban a vehicle."""
    if not verify_session(session_token):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        banned_vehicle = db.query(BannedVehicle).filter(BannedVehicle.vehicle_number == vehicle_number).first()
        if not banned_vehicle:
            raise HTTPException(status_code=404, detail="Vehicle not found in banned list")
        
        db.delete(banned_vehicle)
        db.commit()
        return {"message": f"Vehicle {vehicle_number} unbanned successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error unbanning vehicle: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to unban vehicle")

@admin.put("/api/vehicles/{id}")
async def update_vehicle(
    id: int,
    data: dict,
    session_token: str = Cookie(None),
    db: Session = Depends(get_db)
):
    """Update vehicle log."""
    if not verify_session(session_token):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        vehicle = db.query(VehicleLog).filter(VehicleLog.id == id).first()
        if not vehicle:
            raise HTTPException(status_code=404, detail="Vehicle log not found")
        
        vehicle.driver_name = data.get("driver_name", vehicle.driver_name)
        vehicle.vehicle_type = data.get("vehicle_type", vehicle.vehicle_type)
        vehicle.remarks = data.get("remarks", vehicle.remarks)
        
        db.commit()
        return {"message": f"Vehicle log {id} updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating vehicle: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to update vehicle")

@admin.delete("/api/vehicles/{id}")
async def delete_vehicle(
    id: int,
    session_token: str = Cookie(None),
    db: Session = Depends(get_db)
):
    """Delete vehicle log."""
    if not verify_session(session_token):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        vehicle = db.query(VehicleLog).filter(VehicleLog.id == id).first()
        if not vehicle:
            raise HTTPException(status_code=404, detail="Vehicle log not found")
        
        db.delete(vehicle)
        db.commit()
        return {"message": f"Vehicle log {id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting vehicle: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to delete vehicle")
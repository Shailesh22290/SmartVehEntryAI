#!/bin/bash

echo "========================================"
echo "PlateVision Pro - Starting Service"
echo "========================================"
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Stopping service..."
    kill $APP_PID 2>/dev/null
    echo "Service stopped."
    exit 0
}

# Trap Ctrl+C
trap cleanup INT TERM

echo "Starting PlateVision Pro on port 8000..."
uvicorn app:app --reload --host 0.0.0.0 --port 8000 &
APP_PID=$!

echo ""
echo "========================================"
echo "Service Started!"
echo "========================================"
echo "Staff Interface: http://localhost:8000"
echo "Admin Panel:     http://localhost:8000/admin/login"
echo ""
echo "Admin Credentials:"
echo "  Username: admin"
echo "  Password: admin123"
echo ""
echo "Press Ctrl+C to stop the service..."
echo ""

# Wait for process
wait
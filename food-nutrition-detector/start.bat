@echo off
echo ============================================================
echo Food Nutrition Detector - Startup Script
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Node.js is not installed or not in PATH
    echo Please install Node.js 18 or higher
    pause
    exit /b 1
)

echo Step 1: Checking Python dependencies...
echo.

cd ml-model

REM Check if model exists
if not exist "saved_models\food_classifier.joblib" (
    echo WARNING: Trained model not found!
    echo You need to train the model first.
    echo.
    set /p train="Do you want to train the model now? (yes/no): "
    if /i "%train%"=="yes" (
        echo.
        echo Installing Python dependencies...
        pip install -r requirements.txt
        pip install -r api_requirements.txt
        echo.
        echo Starting training process...
        echo This may take 30-60 minutes with GPU, or 2-4 hours without GPU.
        echo.
        python train.py
    ) else (
        echo.
        echo Please train the model manually:
        echo   cd ml-model
        echo   pip install -r requirements.txt
        echo   pip install -r api_requirements.txt
        echo   python train.py
        pause
        exit /b 1
    )
)

echo.
echo Step 2: Starting Flask API Server...
echo.
start cmd /k "cd /d %cd% && python api_server.py"

timeout /t 3 >nul

cd ..\web-app

echo.
echo Step 3: Checking Node.js dependencies...
echo.

if not exist "node_modules" (
    echo Installing npm packages...
    call npm install
)

echo.
echo Step 4: Starting Next.js Development Server...
echo.
start cmd /k "cd /d %cd% && npm run dev"

echo.
echo ============================================================
echo Both servers are starting!
echo ============================================================
echo.
echo Flask API:  http://localhost:5000
echo Next.js:    http://localhost:3000
echo.
echo Two terminal windows have been opened:
echo   1. Flask API Server (backend)
echo   2. Next.js Dev Server (frontend)
echo.
echo Keep both windows running while using the application.
echo.
echo Press any key to exit this window...
pause >nul

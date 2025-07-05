@echo off
echo 🚀 AI Battle Royale - Docker Deployment
echo ========================================

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Docker is not running. Please start Docker and try again.
    pause
    exit /b 1
)

REM Build the Docker image
echo 📦 Building Docker image...
docker build -t ai-battle-royale .

if errorlevel 1 (
    echo ❌ Error: Docker build failed!
    pause
    exit /b 1
)

echo ✅ Docker image built successfully!

REM Stop and remove existing container if it exists
echo 🧹 Cleaning up existing containers...
docker stop ai-battle-royale >nul 2>&1
docker rm ai-battle-royale >nul 2>&1

REM Run the container
echo 🚀 Starting AI Battle Royale container...
docker run -d --name ai-battle-royale -p 3000:3000 -p 8000:8000 --restart unless-stopped ai-battle-royale

if errorlevel 1 (
    echo ❌ Error: Failed to start container!
    pause
    exit /b 1
)

echo ✅ Container started successfully!
echo.
echo 🌐 Application is now available at:
echo    http://localhost:8000
echo.
echo 📋 Useful commands:
echo    View logs:    docker logs -f ai-battle-royale
echo    Stop app:     docker stop ai-battle-royale
echo    Restart app:  docker restart ai-battle-royale
echo    Remove app:   docker rm -f ai-battle-royale
echo.
echo ⚠️  Remember to add your API keys via environment variables!
echo    You can set them in a .env file or pass them directly to docker run
echo.
pause

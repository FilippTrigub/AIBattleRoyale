# AI Battle Royale - Docker Deployment Guide

This guide explains how to build and run the AI Battle Royale application in a single Docker container.

## 🏗️ Architecture

The Docker container runs both:
- **Backend**: FastAPI server (Python) on port 8000
- **Frontend**: Next.js app (built as static files) served by FastAPI

Both components run in a single container with proper networking configured.

## 📋 Prerequisites

- Docker installed and running
- API keys for the AI providers you want to use

## 🚀 Quick Start

### Option 1: Using the provided scripts

**Windows:**
```bash
run-docker.bat
```

**Linux/Mac:**
```bash
chmod +x run-docker.sh
./run-docker.sh
```

### Option 2: Manual commands

1. **Build the image:**
```bash
docker build -t ai-battle-royale .
```

2. **Run the container:**
```bash
docker run -d --name ai-battle-royale -p 8000:8000 ai-battle-royale
```

3. **Access the application:**
   Open http://localhost:8000 in your browser

## 🔑 Adding API Keys

### Method 1: Environment file
1. Copy the template: `cp .env.template .env`
2. Edit `.env` and add your API keys
3. Run with environment file:
```bash
docker run -d --name ai-battle-royale -p 8000:8000 --env-file .env ai-battle-royale
```

### Method 2: Direct environment variables
```bash
docker run -d \
  --name ai-battle-royale \
  -p 8000:8000 \
  -e OPENAI_API_KEY=your_key_here \
  -e ANTHROPIC_API_KEY=your_key_here \
  ai-battle-royale
```

### Method 3: Docker Compose
```bash
# Set your keys in .env file first
docker-compose up -d
```

## 🎮 How It Works

1. **Build Process:**
   - Stage 1: Builds Next.js frontend into static files
   - Stage 2: Sets up Python backend and copies frontend

2. **Runtime:**
   - FastAPI serves both API endpoints and static frontend files
   - Frontend communicates with backend on the same origin
   - All traffic flows through port 8000

3. **Networking:**
   - Container exposes port 8000
   - API routes: `/start-game`, `/stream-game`, etc.
   - Static files: `/`, `/index.html`, etc.

## 📊 Container Management

### View logs:
```bash
docker logs -f ai-battle-royale
```

### Stop the application:
```bash
docker stop ai-battle-royale
```

### Restart the application:
```bash
docker restart ai-battle-royale
```

### Remove the container:
```bash
docker rm -f ai-battle-royale
```

### Remove the image:
```bash
docker rmi ai-battle-royale
```

## 🔧 Development Mode

For development, you can mount the source code:

```bash
docker run -d \
  --name ai-battle-royale-dev \
  -p 8000:8000 \
  -v $(pwd)/backend:/app/backend \
  --env-file .env \
  ai-battle-royale
```

## 🏥 Health Checks

The container includes a health check that verifies the FastAPI server is running:
```bash
docker health ls
```

## 🐛 Troubleshooting

### Container won't start:
- Check Docker is running: `docker info`
- Check port 8000 is available: `netstat -an | grep 8000`
- View container logs: `docker logs ai-battle-royale`

### Frontend not loading:
- Verify the build succeeded by checking logs during container startup
- Ensure the static files were copied: `docker exec ai-battle-royale ls -la /app/frontend`

### API errors:
- Check that API keys are properly set in environment variables
- Verify network connectivity to AI providers
- Check backend logs for specific error messages

## 📦 Image Details

- **Base Images**: 
  - Frontend builder: `node:18-alpine`
  - Final runtime: `python:3.11-slim`
- **Exposed Port**: 8000
- **Working Directory**: `/app`
- **Health Check**: `curl -f http://localhost:8000/health`

## 🔒 Security Notes

- API keys are passed as environment variables (not stored in image)
- Container runs as non-root user in production setups
- CORS is configured to allow all origins (adjust for production)

## 📈 Performance

The container is optimized for production with:
- Multi-stage build to minimize final image size
- Static file serving through FastAPI
- Health checks for monitoring
- Proper restart policies

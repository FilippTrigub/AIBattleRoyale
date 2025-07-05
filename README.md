# AI Battle Royale - Docker Deployment

This repository contains a full-stack AI Battle Royale application with a FastAPI backend and Next.js frontend, containerized for easy deployment.

## Architecture

- **Frontend**: Next.js React application (static export)
- **Backend**: FastAPI Python server
- **Reverse Proxy**: Nginx 
- **Container**: Single multi-stage Docker container

## Directory Structure

```
AIBattleRoyale/
├── backend/                    # FastAPI backend
│   ├── fastapi_server.py      # Main server file
│   ├── requirements.txt       # Python dependencies
│   └── agent_prompts/         # AI agent prompt files
├── frontend/                  # Next.js frontend
│   ├── app/                   # Next.js app directory
│   ├── components/            # React components
│   ├── package.json          # Node.js dependencies
│   └── next.config.js        # Next.js configuration
├── Dockerfile                # Multi-stage Docker build
├── nginx.conf                # Nginx reverse proxy config
├── start.sh                  # Container startup script
└── .dockerignore            # Docker ignore rules
```

## Quick Start

### Prerequisites

- Docker installed on your system
- Docker Buildkit enabled (for multi-stage builds)

### Build and Run

1. **Navigate to the project directory:**
   ```bash
   cd C:\coding_challanges\AIBattleRoyale\AIBattleRoyale
   ```

2. **Build the Docker image:**
   ```bash
   docker build -t ai-battle-royale .
   ```

3. **Run the container:**
   ```bash
   docker run -p 80:80 ai-battle-royale
   ```

4. **Access the application:**
   - Frontend: http://localhost
   - Backend API: http://localhost/api/
   - Health Check: http://localhost/health

## How It Works

### Multi-Stage Build Process

1. **Stage 1 (frontend-builder)**: 
   - Builds the Next.js application into static files
   - Uses Node.js 18 Alpine image
   - Outputs to `/app/frontend/out`

2. **Stage 2 (backend-dependencies)**:
   - Installs Python dependencies
   - Uses Python 3.10 slim image
   - Prepares backend environment

3. **Stage 3 (production)**:
   - Uses Nginx Alpine as base
   - Installs Python runtime
   - Copies built frontend to Nginx document root
   - Copies backend code and dependencies
   - Configures reverse proxy

### Request Routing

- **Static Files** (`/`): Served directly by Nginx
- **API Requests** (`/api/*`): Proxied to FastAPI backend on port 8000
- **Streaming** (`/api/stream-game/*`): Special handling for Server-Sent Events
- **Health Check** (`/health`): Backend health endpoint

### Services Management

The `start.sh` script manages both services:
1. Starts FastAPI backend on 127.0.0.1:8000
2. Waits for backend to be ready
3. Performs health checks
4. Starts Nginx in foreground
5. Handles graceful shutdown

## Configuration

### Environment Variables

You can set environment variables when running the container:

```bash
docker run -p 80:80 -e OPENAI_API_KEY=your_key ai-battle-royale
```

### API Keys

The application supports multiple AI providers:
- OpenAI (`OPENAI_API_KEY`)
- Anthropic (`ANTHROPIC_API_KEY`) 
- Google (`GOOGLE_API_KEY`)
- Mistral (`MISTRAL_API_KEY`)

## Development vs Production

- **Development**: Frontend connects to `http://localhost:8000`
- **Production**: Frontend uses `/api` (proxied by Nginx)

## Troubleshooting

### Container Logs

View container logs:
```bash
docker logs <container_id>
```

### Backend Logs

Access backend-specific logs inside the container:
```bash
docker exec -it <container_id> cat /var/log/app/backend.log
```

### Service Status

Check if services are running:
```bash
docker exec -it <container_id> ps aux
```

### Port Issues

Ensure port 80 is not in use by other applications:
```bash
netstat -tuln | grep :80
```

## Deployment Options

### Local Development
```bash
docker run -p 80:80 ai-battle-royale
```

### Production Deployment
```bash
docker run -d -p 80:80 --name ai-battle-app --restart unless-stopped ai-battle-royale
```

### Cloud Deployment

The container can be deployed to any platform supporting Docker:
- AWS ECS/Fargate
- Google Cloud Run
- Azure Container Instances
- DigitalOcean App Platform
- Railway
- Fly.io

## Security Notes

- Container runs as non-root user for security
- Nginx includes security headers
- Backend only accessible via reverse proxy
- CORS configured for API endpoints

## Performance

- Nginx efficiently serves static files
- Gzip compression enabled
- Static asset caching configured
- Health checks for monitoring

## Building for Different Platforms

For multi-platform builds:
```bash
docker buildx build --platform linux/amd64,linux/arm64 -t ai-battle-royale .
```

## Support

If you encounter issues:

1. Check Docker version: `docker --version`
2. Verify Buildkit is enabled: `docker buildx version`
3. Review build logs for errors
4. Ensure all required files are present
5. Check container logs for runtime issues

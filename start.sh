#!/bin/sh

echo "🚀 Starting AI Battle Royale Application..."

# Function to check if a port is in use
check_port() {
    netstat -tuln | grep ":$1 " > /dev/null
    return $?
}

# Function to wait for a service to be ready
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local max_attempts=30
    local attempt=1
    
    echo "⏳ Waiting for $service_name to be ready on $host:$port..."
    
    while [ $attempt -le $max_attempts ]; do
        if nc -z $host $port > /dev/null 2>&1; then
            echo "✅ $service_name is ready!"
            return 0
        fi
        
        echo "   Attempt $attempt/$max_attempts - $service_name not ready yet..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo "❌ $service_name failed to start within expected time"
    return 1
}

# Start FastAPI backend in background
echo "🐍 Starting FastAPI backend..."
cd /app/backend

# Create log directory
mkdir -p /var/log/app

# Start FastAPI with proper logging
python3 -u fastapi_server.py > /var/log/app/backend.log 2>&1 &
BACKEND_PID=$!

echo "📝 Backend started with PID: $BACKEND_PID"

# Wait for backend to be ready
if ! wait_for_service "127.0.0.1" "8000" "FastAPI Backend"; then
    echo "❌ Backend failed to start, checking logs..."
    tail -n 20 /var/log/app/backend.log
    exit 1
fi

# Test backend health
echo "🔍 Testing backend health..."
for i in 1 2 3 4 5; do
    if curl -f http://127.0.0.1:8000/health > /dev/null 2>&1; then
        echo "✅ Backend health check passed!"
        break
    elif [ $i -eq 5 ]; then
        echo "❌ Backend health check failed after 5 attempts"
        tail -n 20 /var/log/app/backend.log
        exit 1
    else
        echo "   Health check attempt $i/5 failed, retrying..."
        sleep 2
    fi
done

echo "🌐 Starting Nginx..."

# Test Nginx configuration
nginx -t
if [ $? -ne 0 ]; then
    echo "❌ Nginx configuration test failed"
    exit 1
fi

# Create a function to handle shutdown gracefully
shutdown() {
    echo "🛑 Shutting down services..."
    
    # Stop Nginx gracefully
    echo "   Stopping Nginx..."
    nginx -s quit
    
    # Stop FastAPI backend
    echo "   Stopping FastAPI backend (PID: $BACKEND_PID)..."
    kill -TERM $BACKEND_PID 2>/dev/null
    
    # Wait for processes to stop
    wait $BACKEND_PID 2>/dev/null
    
    echo "✅ All services stopped"
    exit 0
}

# Trap signals for graceful shutdown
trap shutdown TERM INT

# Start Nginx in foreground
echo "✅ All services started successfully!"
echo "🎯 Application ready at http://localhost"
echo "📊 Backend API available at http://localhost/api/"
echo "📈 Health check: http://localhost/health"

# Start nginx and wait for it
nginx -g "daemon off;" &
NGINX_PID=$!

# Wait for either process to exit
wait $NGINX_PID

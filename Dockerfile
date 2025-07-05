# Stage 1: Build Next.js frontend
FROM node:18-alpine AS frontend-builder

WORKDIR /app/frontend

# Install pnpm
RUN npm install -g pnpm

# Copy package files and install dependencies
COPY frontend/package.json frontend/pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile --prod=false

# Copy configuration files needed for build
COPY frontend/next.config.mjs frontend/tailwind.config.ts frontend/tsconfig.json frontend/postcss.config.mjs ./

# Copy frontend source and build
COPY frontend/ ./
RUN pnpm run build --no-lint

# Verify the build output and list contents for debugging
RUN ls -la /app/frontend/
RUN ls -la /app/frontend/out/ || echo "out directory not found, checking .next/"
RUN ls -la /app/frontend/.next/ || echo ".next directory not found"

# Stage 2: Setup Python dependencies
FROM python:3.10-slim AS backend-dependencies

WORKDIR /app

# Install system dependencies for Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 3: Final production stage with Nginx
FROM nginx:alpine AS production

# Install Python and system dependencies
RUN apk add --no-cache \
    python3 \
    py3-pip \
    python3-dev \
    gcc \
    musl-dev \
    linux-headers \
    curl \
    netcat-openbsd \
    bash

# Create app directory
WORKDIR /app

# Copy Python packages from dependencies stage
COPY --from=backend-dependencies /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=backend-dependencies /usr/local/bin /usr/local/bin

# Copy backend source code
COPY backend/ ./backend/

# First copy the frontend build to a temp location
COPY --from=frontend-builder /app/frontend /tmp/frontend-build

# Copy built frontend static files to Nginx document root
# Debug first to see what's available
RUN ls -la /tmp/frontend-build/
RUN if [ -d "/tmp/frontend-build/out" ]; then \
        echo "Found out directory, copying..."; \
        ls -la /tmp/frontend-build/out/; \
        cp -r /tmp/frontend-build/out/* /usr/share/nginx/html/; \
    elif [ -d "/tmp/frontend-build/.next" ]; then \
        echo "Found .next directory, copying..."; \
        ls -la /tmp/frontend-build/.next/; \
        cp -r /tmp/frontend-build/.next/* /usr/share/nginx/html/; \
    else \
        echo "No build output found! Available directories:"; \
        find /tmp/frontend-build -type d -name "*build*" -o -name "*out*" -o -name "*dist*" -o -name "*.next*"; \
        exit 1; \
    fi

# Copy Nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Create startup script
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Create non-root user for security
RUN addgroup -g 1001 -S appuser && \
    adduser -S appuser -u 1001 -G appuser

# Change ownership of necessary directories
RUN chown -R appuser:appuser /app && \
    chown appuser:appuser /start.sh

# Expose port 80
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost/ || exit 1

# Start services
CMD ["/start.sh"]

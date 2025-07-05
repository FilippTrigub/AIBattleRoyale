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
RUN echo "Build completed - checking .next directory:"
RUN ls -la /app/frontend/.next/ || echo ".next directory not found - build may have failed"

# Stage 2: Final production stage with Node.js + Python + Nginx
FROM node:18-alpine AS production

# Install Python, Nginx and system dependencies
RUN apk add --no-cache \
    python3 \
    py3-pip \
    python3-dev \
    gcc \
    musl-dev \
    linux-headers \
    curl \
    netcat-openbsd \
    bash \
    nginx

# Create app directory
WORKDIR /app

# Copy backend requirements and install them in a virtual environment
COPY backend/requirements.txt ./
RUN python3 -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Add virtual environment to PATH
ENV PATH="/opt/venv/bin:$PATH"

# Copy backend source code
COPY backend/ ./backend/

# Copy frontend build and install production dependencies
COPY --from=frontend-builder /app/frontend/.next ./frontend/.next/
COPY --from=frontend-builder /app/frontend/package.json ./frontend/
COPY --from=frontend-builder /app/frontend/public ./frontend/public/
COPY --from=frontend-builder /app/frontend/next.config.mjs ./frontend/

# Install only production dependencies for Next.js runtime
WORKDIR /app/frontend
RUN npm install --omit=dev --silent

# Return to app directory
WORKDIR /app

# Verify the frontend build
RUN ls -la /app/frontend/.next/ && \
    echo "Frontend build copied successfully"

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

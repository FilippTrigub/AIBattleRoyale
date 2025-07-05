/** @type {import('next').NextConfig} */
const nextConfig = {
  // Use standard production build (not static export)
  // output: 'export', // Removed - incompatible with SSE and dynamic features
  
  // Add trailing slash to URLs for better compatibility
  trailingSlash: true,
  
  // Optimize images for production
  images: {
    unoptimized: true
  },
  
  // Configure asset prefix for production
  assetPrefix: process.env.NODE_ENV === 'production' ? '' : '',
  
  // Use default .next directory
  // distDir: 'out', // Removed - using default .next
  
  // Build configuration
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  
  // Disable experimental features that might cause issues
  experimental: {
    esmExternals: false,
  },
}

export default nextConfig

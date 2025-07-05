/** @type {import('next').NextConfig} */
const nextConfig = {
  // Enable static HTML export
  output: 'export',
  
  // Add trailing slash to URLs for better static hosting compatibility
  trailingSlash: true,
  
  // Optimize images for static export
  images: {
    unoptimized: true
  },
  
  // Configure asset prefix for production
  assetPrefix: process.env.NODE_ENV === 'production' ? '' : '',
  
  // Configure build output directory
  distDir: 'out',
  
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

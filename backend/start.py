#!/usr/bin/env python3
"""
MisrBase Backend Startup Script

This script initializes the backend server with proper model loading and error handling.
"""

import os
import sys
import logging
from app import app, load_models
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main startup function"""
    print("🚀 Starting MisrBase Backend Server...")
    print("=" * 50)
    
    # Check if we're in development or production
    debug_mode = Config.DEBUG
    print(f"Debug mode: {debug_mode}")
    
    # Create models directory if it doesn't exist
    os.makedirs(Config.MODELS_DIR, exist_ok=True)
    print(f"Models directory: {Config.MODELS_DIR}")
    
    # Load models
    print("\n📦 Loading AI Models...")
    try:
        models_loaded = load_models()
        if models_loaded:
            print("✅ All models loaded successfully!")
        else:
            print("⚠️  Some models failed to load. Check the logs above.")
            print("   The server will start but prediction endpoints may not work.")
    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")
        print("   The server will start but prediction endpoints will not work.")
        print("   Please check your model configuration and try again.")
    
    # Print configuration summary
    print("\n⚙️  Configuration Summary:")
    print(f"   - Device: {Config.DEVICE}")
    print(f"   - Max sequence length: {Config.MAX_SEQUENCE_LENGTH}")
    print(f"   - CORS origins: {Config.CORS_ORIGINS}")
    
    # Start the server
    print("\n🌐 Starting Flask server...")
    print(f"   - URL: http://localhost:5000")
    print(f"   - Health check: http://localhost:5000/api/health")
    print("\n" + "=" * 50)
    
    try:
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=debug_mode,
            use_reloader=False  # Disable reloader to avoid duplicate model loading
        )
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 
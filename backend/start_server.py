#!/usr/bin/env python3
"""
Startup script for MisrBase Backend API
"""

import os
import sys
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backend.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main startup function"""
    
    print("🚀 Starting MisrBase Backend API Server...")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    try:
        # Import and run the Flask app
        from app import app, check_api_status
        
        # Check API status
        logger.info("Checking Hugging Face API access...")
        api_accessible = check_api_status()
        
        if api_accessible:
            logger.info("✅ Hugging Face API is accessible!")
            print("✅ Server ready to accept requests!")
        else:
            logger.warning("⚠️  Hugging Face API may not be accessible. Server will start but predictions may fail.")
            print("⚠️  Server starting - check internet connection for predictions.")
        
        # Start the server
        logger.info("Starting Flask server...")
        app.run(
            host='0.0.0.0', 
            port=5000, 
            debug=False,  # Set to False for production
            threaded=True
        )
        
    except ImportError as e:
        logger.error(f"❌ Import error: {str(e)}")
        print(f"❌ Failed to import required modules: {str(e)}")
        print("Please ensure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"❌ Startup error: {str(e)}")
        print(f"❌ Failed to start server: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
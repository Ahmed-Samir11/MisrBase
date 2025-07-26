#!/usr/bin/env python3
"""
Test script to verify Hugging Face API access
"""

import requests
import os
import json

def test_huggingface_api():
    """Test accessing models via Hugging Face API"""
    
    print("🧪 Testing Hugging Face API Access...")
    print("=" * 50)
    
    # API configuration
    HUGGINGFACE_API_TOKEN = os.getenv('HUGGINGFACE_API_TOKEN', '')
    HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models"
    
    # Model endpoints
    MODELS = {
        'category': 'MohammedHany123/category_model',
        'misconception': 'MohammedHany123/misconception_model',
        'correctness': 'MohammedHany123/correct_model'
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    if HUGGINGFACE_API_TOKEN:
        headers["Authorization"] = f"Bearer {HUGGINGFACE_API_TOKEN}"
        print("🔑 Using API token for authentication")
    else:
        print("🌐 Using public access (no API token)")
    
    test_text = "I think the answer is 5 because 2 + 3 = 5"
    
    # Test each model
    for model_type, model_name in MODELS.items():
        try:
            print(f"\n📦 Testing {model_type} model...")
            
            payload = {
                "inputs": test_text,
                "options": {
                    "wait_for_model": True
                }
            }
            
            response = requests.post(
                f"{HUGGINGFACE_API_URL}/{model_name}",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ {model_type} model API call successful!")
                
                if isinstance(result, list) and len(result) > 0:
                    prediction_data = result[0]
                    if 'label' in prediction_data:
                        print(f"   Prediction: {prediction_data['label']}")
                        if 'score' in prediction_data:
                            print(f"   Confidence: {prediction_data['score']:.4f}")
                    else:
                        print(f"   Raw response: {prediction_data}")
                else:
                    print(f"   Raw response: {result}")
                    
            else:
                print(f"❌ API Error: {response.status_code}")
                print(f"   Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Error testing {model_type} model: {str(e)}")
    
    print("\n" + "=" * 50)
    print("🎉 API testing completed!")
    
    # Test the backend's API check function
    print("\n🔍 Testing backend API check...")
    try:
        from app import check_api_status
        api_accessible = check_api_status()
        if api_accessible:
            print("✅ Backend API check: PASSED")
        else:
            print("❌ Backend API check: FAILED")
    except Exception as e:
        print(f"❌ Error importing backend: {str(e)}")

if __name__ == "__main__":
    test_huggingface_api() 
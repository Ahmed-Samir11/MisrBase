from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import re
import requests
import json
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Global variables for API configuration
HUGGINGFACE_API_TOKEN = os.getenv('HUGGINGFACE_API_TOKEN', '')
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models"

# Model endpoints
MODELS = {
    'category': 'MohammedHany123/category_model',
    'misconception': 'MohammedHany123/misconception_model',
    'correctness': 'MohammedHany123/correct_model'
}

def extract_mathematical_features(text): 
    """Extract mathematical and linguistic features from text""" 
    features = {} 
     
    # Basic text features 
    features['length'] = len(str(text)) 
    features['word_count'] = len(str(text).split()) 
    features['sentence_count'] = len(str(text).split('.')) 
     
    # Mathematical content features 
    features['has_fraction'] = int(bool(re.search(r'\\frac|/', str(text)))) 
    features['has_decimal'] = int(bool(re.search(r'\d+\.\d+', str(text)))) 
    features['has_negative'] = int(bool(re.search(r'-\d+', str(text)))) 
    features['has_multiplication'] = int(bool(re.search(r'\*|×|times', str(text)))) 
    features['has_division'] = int(bool(re.search(r'÷|div|/', str(text)))) 
    features['has_addition'] = int(bool(re.search(r'\+|plus|add', str(text)))) 
    features['has_subtraction'] = int(bool(re.search(r'-|minus|subtract', str(text)))) 
    features['has_equals'] = int(bool(re.search(r'=|equals', str(text)))) 
     
    # Count numbers 
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', str(text)) 
    features['number_count'] = len(numbers) 
     
    # Reasoning indicators 
    features['has_because'] = int('because' in str(text).lower()) 
    features['has_so'] = int('so' in str(text).lower()) 
    features['has_therefore'] = int('therefore' in str(text).lower()) 
    features['has_since'] = int('since' in str(text).lower()) 
     
    # Uncertainty indicators 
    features['has_think'] = int('think' in str(text).lower()) 
    features['has_maybe'] = int('maybe' in str(text).lower()) 
    features['has_probably'] = int('probably' in str(text).lower()) 
    features['has_guess'] = int('guess' in str(text).lower()) 
     
    # Common misconception keywords 
    features['mentions_bigger'] = int('bigger' in str(text).lower()) 
    features['mentions_smaller'] = int('smaller' in str(text).lower()) 
    features['mentions_more'] = int('more' in str(text).lower()) 
    features['mentions_less'] = int('less' in str(text).lower()) 
     
    return features 

def call_huggingface_api(model_name, text, api_token=None):
    """Call Hugging Face inference API for a model"""
    
    headers = {
        "Content-Type": "application/json"
    }
    
    # Add authorization header if token is provided
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"
    
    payload = {
        "inputs": text,
        "options": {
            "wait_for_model": True
        }
    }
    
    try:
        response = requests.post(
            f"{HUGGINGFACE_API_URL}/{model_name}",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"API Error for {model_name}: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"Request failed for {model_name}: {str(e)}")
        return None

def predict_with_huggingface_api(texts, model_name):
    """Make predictions using Hugging Face API"""
    
    predictions = []
    probabilities = []
    
    for text in texts:
        result = call_huggingface_api(model_name, text, HUGGINGFACE_API_TOKEN)
        
        if result and isinstance(result, list) and len(result) > 0:
            # Extract prediction and probabilities
            prediction_data = result[0]
            
            if 'label' in prediction_data:
                # Classification result
                pred_class = int(prediction_data['label'].split('_')[-1]) if '_' in prediction_data['label'] else 0
                predictions.append(pred_class)
                
                # Get probabilities if available
                if 'score' in prediction_data:
                    # Single score - create dummy probabilities
                    probs = [0.0] * 10  # Assume max 10 classes
                    probs[pred_class] = prediction_data['score']
                    probabilities.append(probs)
                else:
                    probabilities.append([1.0] + [0.0] * 9)  # Dummy probabilities
            else:
                # Fallback
                predictions.append(0)
                probabilities.append([1.0] + [0.0] * 9)
        else:
            # Fallback for failed requests
            predictions.append(0)
            probabilities.append([1.0] + [0.0] * 9)
    
    return np.array(predictions), np.array(probabilities)

def get_top_k_predictions(probabilities, k=3):
    """Get top-k predictions from probability arrays"""
    return np.argsort(probabilities, axis=1)[:, -k:][:, ::-1]

def predict_top3_api(texts):
    """Generate top-3 predictions using Hugging Face API"""
    
    preds = []
    
    print("Predicting category...")
    cat_preds, cat_probs = predict_with_huggingface_api(texts, MODELS['category'])
    
    print("Predicting misconception...")
    misc_preds, misc_probs = predict_with_huggingface_api(texts, MODELS['misconception'])
    
    print("Predicting correctness...")
    corr_preds, corr_probs = predict_with_huggingface_api(texts, MODELS['correctness'])
    
    # Get top-3 for each component
    top3_cat = get_top_k_predictions(cat_probs, k=3)
    top3_misc = get_top_k_predictions(misc_probs, k=3)
    
    for i in range(len(texts)):
        sample_preds = []
        
        # Combine predictions to create top-3 overall predictions
        correctness = 'True' if corr_preds[i] == 1 else 'False'
        
        # Create combinations of top predictions
        for cat_idx in top3_cat[i]:
            for misc_idx in top3_misc[i]:
                if len(sample_preds) >= 3:
                    break
                
                # Use class indices as fallback names
                cat_name = f"Category_{cat_idx}"
                misc_name = f"Misconception_{misc_idx}"
                
                prediction = f"{correctness}_{cat_name}:{misc_name}"
                if prediction not in sample_preds:
                    sample_preds.append(prediction)
            
            if len(sample_preds) >= 3:
                break
        
        # Ensure we always have 3 predictions
        while len(sample_preds) < 3:
            sample_preds.append(sample_preds[0] if sample_preds else "False_Neither:NA")
        
        preds.append(sample_preds[:3])
    
    return preds

def check_api_status():
    """Check if Hugging Face API is accessible"""
    try:
        # Test with a simple request to one of the models
        result = call_huggingface_api(MODELS['category'], "test", HUGGINGFACE_API_TOKEN)
        return result is not None
    except:
        return False

# Mock teacher database (in production, use a real database)
TEACHERS_DB = {
    "teacher@misrbase.edu": {
        "password": "password123",
        "name": "Ahmed Teacher",
        "school": "MisrBase Academy"
    }
}

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    api_status = check_api_status()
    return jsonify({
        "status": "healthy",
        "message": "MisrBase API is running",
        "huggingface_api_accessible": api_status,
        "models": list(MODELS.keys())
    })

@app.route('/api/auth/signin', methods=['POST'])
def teacher_signin():
    """Teacher sign-in endpoint"""
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            return jsonify({
                "success": False,
                "message": "Email and password are required"
            }), 400
        
        # Check credentials (in production, use proper authentication)
        if email in TEACHERS_DB and TEACHERS_DB[email]['password'] == password:
            return jsonify({
                "success": True,
                "message": "Sign in successful",
                "teacher": {
                    "email": email,
                    "name": TEACHERS_DB[email]['name'],
                    "school": TEACHERS_DB[email]['school']
                }
            })
        else:
            return jsonify({
                "success": False,
                "message": "Invalid email or password"
            }), 401
            
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error during sign in: {str(e)}"
        }), 500

@app.route('/api/predict', methods=['POST'])
def predict_misconception():
    """Predict misconception from teacher input"""
    try:
        data = request.get_json()
        question_text = data.get('QuestionText', '')
        mc_answer = data.get('MC_Answer', '')
        student_explanation = data.get('StudentExplanation', '')
        
        if not question_text or not mc_answer or not student_explanation:
            return jsonify({
                "success": False,
                "message": "QuestionText, MC_Answer, and StudentExplanation are required"
            }), 400
        
        # Check if API is accessible
        if not check_api_status():
            return jsonify({
                "success": False,
                "message": "Hugging Face API is not accessible. Please check your internet connection."
            }), 503
        
        # Make predictions using API
        predictions = predict_top3_api([student_explanation])
        
        # Format results
        top3_predictions = predictions[0] if predictions else ["False_Neither:NA"]
        
        # Parse predictions for better understanding
        parsed_predictions = []
        for pred in top3_predictions:
            try:
                correctness, rest = pred.split('_', 1)
                category, misconception = rest.split(':', 1)
                
                parsed_predictions.append({
                    "full_prediction": pred,
                    "correctness": correctness,
                    "category": category,
                    "misconception": misconception,
                    "confidence": "high" if pred == top3_predictions[0] else "medium" if pred == top3_predictions[1] else "low"
                })
            except:
                parsed_predictions.append({
                    "full_prediction": pred,
                    "correctness": "Unknown",
                    "category": "Unknown",
                    "misconception": "Unknown",
                    "confidence": "low"
                })
        
        return jsonify({
            "success": True,
            "predictions": parsed_predictions,
            "top_prediction": top3_predictions[0],
            "analysis": {
                "question_length": len(question_text),
                "answer_length": len(mc_answer),
                "explanation_length": len(student_explanation),
                "has_mathematical_content": bool(re.search(r'\d+', student_explanation))
            }
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error during prediction: {str(e)}"
        }), 500

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint for multiple student responses"""
    try:
        data = request.get_json()
        responses = data.get('responses', [])
        
        if not responses:
            return jsonify({
                "success": False,
                "message": "Responses array is required"
            }), 400
        
        # Check if API is accessible
        if not check_api_status():
            return jsonify({
                "success": False,
                "message": "Hugging Face API is not accessible. Please check your internet connection."
            }), 503
        
        # Extract student explanations
        explanations = [response.get('StudentExplanation', '') for response in responses]
        
        # Make predictions using API
        predictions = predict_top3_api(explanations)
        
        # Format results
        results = []
        for i, preds in enumerate(predictions):
            parsed_preds = []
            for pred in preds:
                try:
                    correctness, rest = pred.split('_', 1)
                    category, misconception = rest.split(':', 1)
                    parsed_preds.append({
                        "full_prediction": pred,
                        "correctness": correctness,
                        "category": category,
                        "misconception": misconception
                    })
                except:
                    parsed_preds.append({
                        "full_prediction": pred,
                        "correctness": "Unknown",
                        "category": "Unknown",
                        "misconception": "Unknown"
                    })
            
            results.append({
                "row_id": i,
                "predictions": parsed_preds,
                "top_prediction": preds[0] if preds else "False_Neither:NA"
            })
        
        return jsonify({
            "success": True,
            "results": results,
            "total_processed": len(results)
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error during batch prediction: {str(e)}"
        }), 500

if __name__ == '__main__':
    # Start the server
    print("🚀 Starting MisrBase API Server with Hugging Face API...")
    print("=" * 60)
    print("Models will be accessed directly from Hugging Face API")
    print("No local model downloads required!")
    print("=" * 60)
    
    # Check API status
    if check_api_status():
        print("✅ Hugging Face API is accessible!")
    else:
        print("⚠️  Warning: Hugging Face API may not be accessible")
        print("   Make sure you have internet connection")
        if HUGGINGFACE_API_TOKEN:
            print("   API token is configured")
        else:
            print("   No API token configured - using public access")
    
    print("✅ Server ready to accept requests!")
    
    # Run the server
    app.run(host='0.0.0.0', port=5000, debug=True) 
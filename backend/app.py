from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import re
import torch
import joblib
import os
import json
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Global variables for models and encoders
models = {}
encoders = {}
tfidf_vectorizer = None
question_encoder = None
answer_encoder = None
feature_names = []

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

def prepare_features(df, tfidf_vectorizer, question_encoder, answer_encoder):
    """Prepare all features for modeling""" 
    print("=== Feature Engineering ===") 
     
    # Extract mathematical features 
    math_features = [] 
    for explanation in df['StudentExplanation']: 
        math_features.append(extract_mathematical_features(explanation)) 
     
    math_features_df = pd.DataFrame(math_features) 
    print(f"Mathematical features created: {math_features_df.shape[1]}") 
     
    # TF-IDF features for text 
    tfidf_matrix = tfidf_vectorizer.transform(df['StudentExplanation'].astype(str))
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])])
    print(f"TF-IDF features created: {tfidf_df.shape[1]}") 
          
    question_features = pd.DataFrame({ 
        'question_encoded': question_encoder.transform(df['QuestionText'].astype(str)),
        'answer_encoded': answer_encoder.transform(df['MC_Answer'].astype(str)), 
        'question_length': df['QuestionText'].str.len(), 
        'answer_length': df['MC_Answer'].str.len() 
    }) 
    print(f"Question/Answer features created: {question_features.shape[1]}") 
     
    # Combine all features 
    features_df = pd.concat([math_features_df, tfidf_df, question_features], axis=1) 
    print(f"Total features: {features_df.shape[1]}") 
     
    return features_df

def predict_with_roberta(model, tokenizer, texts, device='cpu'):
    """Make predictions with trained RoBERTa model"""
    
    model.eval()
    model.to(device)
    
    predictions = []
    probabilities = []
    
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                str(text),
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            ).to(device)
            
            outputs = model(**inputs)
            logits = outputs.logits
            
            probs = torch.softmax(logits, dim=-1)
            pred = torch.argmax(logits, dim=-1)
            
            predictions.append(pred.cpu().numpy()[0])
            probabilities.append(probs.cpu().numpy()[0])
    
    return np.array(predictions), np.array(probabilities)

def get_top_k_predictions(probabilities, k=3):
    """Get top-k predictions from probability arrays"""
    return np.argsort(probabilities, axis=1)[:, -k:][:, ::-1]

def predict_top3_roberta(models, encoders, df, X):
    """Generate top-3 predictions using trained models"""
    
    preds = []
    texts = df['StudentExplanation'].tolist()
    
    # Get predictions from each model
    print("Predicting lightGBM...")
    X_val = X.loc[df.index]
    corr_preds = models['correctness'].predict(X_val)

    print("Predicting Roberta (category)...")
    cat_preds, cat_probs = predict_with_roberta(
        models['category'], models['category_tokenizer'], texts
    )

    print("Predicting Roberta (misconception)...")
    misc_preds, misc_probs = predict_with_roberta(
        models['misconception'], models['misconception_tokenizer'], texts
    )
    
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
                
                cat_name = encoders['category'].inverse_transform([cat_idx])[0]
                misc_name = encoders['misconception'].inverse_transform([misc_idx])[0]
                
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

def load_models():
    """Load all models and encoders from Hugging Face and local files"""
    global models, encoders, tfidf_vectorizer, question_encoder, answer_encoder
    
    try:
        print("Loading models from Hugging Face...")
        
        # Load models from Hugging Face (replace with your actual model names)
        # You'll need to upload your trained models to Hugging Face first
        category_model_name = "your-username/misrbase-category-model"  # Replace with actual
        misconception_model_name = "your-username/misrbase-misconception-model"  # Replace with actual
        
        # Load RoBERTa models
        models['category'] = RobertaForSequenceClassification.from_pretrained(category_model_name)
        models['category_tokenizer'] = RobertaTokenizer.from_pretrained(category_model_name)
        
        models['misconception'] = RobertaForSequenceClassification.from_pretrained(misconception_model_name)
        models['misconception_tokenizer'] = RobertaTokenizer.from_pretrained(misconception_model_name)
        
        # Load LightGBM model (you'll need to save this separately)
        models['correctness'] = lgb.Booster(model_file='models/correctness_model.txt')
        
        # Load encoders and vectorizers
        encoders['category'] = joblib.load('models/category_encoder.joblib')
        encoders['misconception'] = joblib.load('models/misconception_encoder.joblib')
        tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.joblib')
        question_encoder = joblib.load('models/question_encoder.joblib')
        answer_encoder = joblib.load('models/answer_encoder.joblib')
        
        print("✅ All models loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")
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
    return jsonify({
        "status": "healthy",
        "message": "MisrBase API is running",
        "models_loaded": len(models) > 0
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
        
        # Check if models are loaded
        if not models:
            return jsonify({
                "success": False,
                "message": "Models not loaded. Please try again later."
            }), 503
        
        # Create DataFrame for prediction
        df = pd.DataFrame({
            'QuestionText': [question_text],
            'MC_Answer': [mc_answer],
            'StudentExplanation': [student_explanation]
        })
        
        # Prepare features
        X = prepare_features(df, tfidf_vectorizer, question_encoder, answer_encoder)
        
        # Make predictions
        predictions = predict_top3_roberta(models, encoders, df, X)
        
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
        
        # Check if models are loaded
        if not models:
            return jsonify({
                "success": False,
                "message": "Models not loaded. Please try again later."
            }), 503
        
        # Create DataFrame for batch prediction
        df_data = []
        for i, response in enumerate(responses):
            df_data.append({
                'row_id': i,
                'QuestionText': response.get('QuestionText', ''),
                'MC_Answer': response.get('MC_Answer', ''),
                'StudentExplanation': response.get('StudentExplanation', '')
            })
        
        df = pd.DataFrame(df_data)
        
        # Prepare features
        X = prepare_features(df, tfidf_vectorizer, question_encoder, answer_encoder)
        
        # Make predictions
        predictions = predict_top3_roberta(models, encoders, df, X)
        
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
    # Load models on startup
    print("🚀 Starting MisrBase API Server...")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Load models
    models_loaded = load_models()
    
    if models_loaded:
        print("✅ Server ready to accept requests!")
    else:
        print("⚠️  Server starting without models. Some endpoints may not work.")
    
    # Run the server
    app.run(host='0.0.0.0', port=5000, debug=True) 
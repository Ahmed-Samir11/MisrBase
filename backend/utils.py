import pandas as pd
import numpy as np
import re
import torch
import joblib
import os
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

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

def parse_prediction(prediction_string):
    """Parse a prediction string into components"""
    try:
        correctness, rest = prediction_string.split('_', 1)
        category, misconception = rest.split(':', 1)
        
        return {
            "full_prediction": prediction_string,
            "correctness": correctness,
            "category": category,
            "misconception": misconception
        }
    except:
        return {
            "full_prediction": prediction_string,
            "correctness": "Unknown",
            "category": "Unknown",
            "misconception": "Unknown"
        }

def analyze_student_response(question_text, mc_answer, student_explanation):
    """Analyze a student response and return insights"""
    analysis = {
        "question_length": len(question_text),
        "answer_length": len(mc_answer),
        "explanation_length": len(student_explanation),
        "has_mathematical_content": bool(re.search(r'\d+', student_explanation)),
        "has_fractions": bool(re.search(r'\\frac|/', student_explanation)),
        "has_decimals": bool(re.search(r'\d+\.\d+', student_explanation)),
        "has_negative_numbers": bool(re.search(r'-\d+', student_explanation)),
        "has_operators": bool(re.search(r'[\+\-\*/=]', student_explanation)),
        "reasoning_indicators": {
            "has_because": 'because' in student_explanation.lower(),
            "has_so": 'so' in student_explanation.lower(),
            "has_therefore": 'therefore' in student_explanation.lower(),
            "has_since": 'since' in student_explanation.lower()
        },
        "uncertainty_indicators": {
            "has_think": 'think' in student_explanation.lower(),
            "has_maybe": 'maybe' in student_explanation.lower(),
            "has_probably": 'probably' in student_explanation.lower(),
            "has_guess": 'guess' in student_explanation.lower()
        }
    }
    
    return analysis 
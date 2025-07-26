import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration class for MisrBase backend"""
    
    # Flask settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'misrbase-secret-key-change-in-production')
    DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
    
    # Model settings
    MODELS_DIR = os.getenv('MODELS_DIR', 'models')
    
    # Hugging Face model names (replace with your actual model names)
    CATEGORY_MODEL_NAME = os.getenv('CATEGORY_MODEL_NAME', 'your-username/misrbase-category-model')
    MISCONCEPTION_MODEL_NAME = os.getenv('MISCONCEPTION_MODEL_NAME', 'your-username/misrbase-misconception-model')
    
    # Local model files
    CORRECTNESS_MODEL_PATH = os.path.join(MODELS_DIR, 'correctness_model.txt')
    CATEGORY_ENCODER_PATH = os.path.join(MODELS_DIR, 'category_encoder.joblib')
    MISCONCEPTION_ENCODER_PATH = os.path.join(MODELS_DIR, 'misconception_encoder.joblib')
    TFIDF_VECTORIZER_PATH = os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib')
    QUESTION_ENCODER_PATH = os.path.join(MODELS_DIR, 'question_encoder.joblib')
    ANSWER_ENCODER_PATH = os.path.join(MODELS_DIR, 'answer_encoder.joblib')
    
    # API settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', 'http://localhost:3000').split(',')
    
    # Prediction settings
    MAX_SEQUENCE_LENGTH = 512
    DEVICE = 'cuda' if os.getenv('USE_CUDA', 'False').lower() == 'true' else 'cpu'
    
    # Authentication settings (for production, use proper auth)
    TEACHER_CREDENTIALS = {
        "teacher@misrbase.edu": {
            "password": "password123",
            "name": "Ahmed Teacher",
            "school": "MisrBase Academy"
        }
    } 
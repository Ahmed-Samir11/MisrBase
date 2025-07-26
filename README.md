# MisrBase: Math Insight & Strategy Building

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![React](https://img.shields.io/badge/React-18+-blue.svg)](https://reactjs.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com/)

> Revolutionizing Curriculum Quality through AI-Powered Educational Diagnostics

## 🎯 Vision

MisrBase leverages advanced Arabic Natural Language Processing (NLP) and machine learning specifically trained on the Egyptian curriculum and local student language patterns. Students solve problems and crucially, write or speak their reasoning steps. The platform analyzes this text/speech to:

- **Identify Misconception Patterns**: Detects specific errors in logic (e.g., misunderstanding fractions as whole numbers, misapplying algebraic rules common in Egyptian classrooms).
- **Categorize Errors**: Links misconceptions to specific curriculum topics and learning objectives.

## 🏗️ Architecture

```
MisrBase/
├── src/                    # React Frontend
│   ├── components/         # Reusable UI components
│   ├── views/             # Page components
│   └── assets/            # Static assets
├── backend/               # Flask API Server
│   ├── app.py            # Main API server
│   ├── utils.py          # ML utilities
│   ├── config.py         # Configuration
│   └── requirements.txt  # Python dependencies
├── public/               # Public assets
└── Documentation/        # Project documentation
```

## 🚀 Quick Start

### Prerequisites

- **Node.js** (v16 or higher)
- **Python** (3.8 or higher)
- **Git**

### Frontend Setup

```bash
# Install dependencies
npm install

# Start development server
npm start
```

The React app will be available at `http://localhost:3000`

### Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp env.example .env

# Edit .env with your configuration (optional)
# The backend automatically loads models from Hugging Face

# Start the server
python start_server.py
```

The API will be available at `http://localhost:5000`

## 📱 Features

### For Teachers
- **Real-time Analysis**: Immediate misconception detection from student responses
- **Personalized Dashboards**: Class-level and individual student insights
- **Targeted Remediation**: Suggested activities aligned with MoE guidelines
- **Progress Tracking**: Monitor improvement over time

### For Ministry of Education
- **National Analytics**: Aggregated, anonymized data revealing systemic weaknesses
- **Curriculum Insights**: Data-driven evidence for curriculum refinements
- **Regional Comparisons**: Governorate-level performance analysis
- **Policy Support**: Evidence-based decision making

## 🔧 Technology Stack

### Frontend
- **React 18** - Modern UI framework
- **Reactstrap** - Bootstrap 4 components for React
- **React Router** - Client-side routing
- **Axios** - HTTP client for API calls

### Backend
- **Flask** - Python web framework
- **Transformers** - Hugging Face ML library
- **PyTorch** - Deep learning framework
- **LightGBM** - Gradient boosting framework
- **scikit-learn** - Machine learning utilities

### AI/ML
- **RoBERTa** - Pre-trained language models for classification
- **TF-IDF** - Text feature extraction
- **MAP@3** - Optimized evaluation metric
- **Feature Engineering** - Mathematical content analysis

## 📊 API Endpoints

### Authentication
```http
POST /api/auth/signin
Content-Type: application/json

{
  "email": "teacher@misrbase.edu",
  "password": "password123"
}
```

### Prediction
```http
POST /api/predict
Content-Type: application/json

{
  "QuestionText": "What is 1/2 + 1/4?",
  "MC_Answer": "3/4",
  "StudentExplanation": "I added the numerators and denominators"
}
```

### Health Check
```http
GET /api/health
```

## 🎨 UI Components

### Teacher Sign-In
- Secure authentication interface
- Professional design with animations
- Error handling and validation

### Analysis Dashboard
- Real-time prediction display
- Confidence scoring
- Detailed insights breakdown
- Mathematical content analysis

## 🔒 Security

- **CORS Configuration**: Proper cross-origin resource sharing
- **Input Validation**: Comprehensive data validation
- **Error Handling**: Graceful error management
- **Environment Variables**: Secure configuration management

## 📈 Performance

- **MAP@3 Optimization**: Top-3 predictions for better accuracy
- **Batch Processing**: Efficient handling of multiple responses
- **Caching**: Model loading optimization
- **Async Processing**: Non-blocking API responses

## 🚀 Deployment

### Frontend Deployment
```bash
# Build for production
npm run build

# Deploy to your preferred platform
# (Netlify, Vercel, AWS S3, etc.)
```

### Backend Deployment
```bash
# Using Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Using Docker
docker build -t misrbase-backend .
docker run -p 5000:5000 misrbase-backend
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## 🙏 Acknowledgments

- **Egypt Vision 2030** - Aligned with national education goals
- **Ministry of Education** - Supporting educational innovation
- **Creative Tim** - React template design system
- **Hugging Face** - Pre-trained language models
- **Open Source Community** - Contributing to educational technology
---

**MisrBase** - Building Egypt's Future through AI-Powered Education 🎓🇪🇬

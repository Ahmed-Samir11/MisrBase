# MisrBase Backend API

This is the backend API for MisrBase, an AI-powered educational diagnostic platform for identifying student misconceptions in mathematics.

## Features

- **Teacher Authentication**: Secure sign-in for teachers
- **Real-time Analysis**: Instant misconception prediction from student responses
- **Batch Processing**: Handle multiple student responses at once
- **Comprehensive Insights**: Detailed analysis of student explanations
- **MAP@3 Optimization**: Top-3 predictions for better accuracy

## API Endpoints

### Authentication
- `POST /api/auth/signin` - Teacher sign-in

### Prediction
- `POST /api/predict` - Single student response analysis
- `POST /api/batch-predict` - Batch analysis of multiple responses

### Health Check
- `GET /api/health` - API health status

## Setup Instructions

### 1. Prerequisites

- Python 3.8+
- pip
- Git

### 2. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Model Setup

#### Option A: Using Hugging Face Models (Recommended)

1. Upload your trained models to Hugging Face:
   - Category classification model
   - Misconception classification model

2. Update the model names in `config.py` or set environment variables:
   ```bash
   export CATEGORY_MODEL_NAME="your-username/misrbase-category-model"
   export MISCONCEPTION_MODEL_NAME="your-username/misrbase-misconception-model"
   ```

#### Option B: Using Local Models

1. Create a `models/` directory:
   ```bash
   mkdir models
   ```

2. Place your model files in the `models/` directory:
   - `correctness_model.txt` (LightGBM model)
   - `category_encoder.joblib`
   - `misconception_encoder.joblib`
   - `tfidf_vectorizer.joblib`
   - `question_encoder.joblib`
   - `answer_encoder.joblib`

### 4. Environment Configuration

1. Copy the example environment file:
   ```bash
   cp env.example .env
   ```

2. Edit `.env` with your configuration:
   ```env
   SECRET_KEY=your-secret-key-here
   DEBUG=True
   CATEGORY_MODEL_NAME=your-username/misrbase-category-model
   MISCONCEPTION_MODEL_NAME=your-username/misrbase-misconception-model
   CORS_ORIGINS=http://localhost:3000
   USE_CUDA=False
   ```

### 5. Running the Server

#### Development Mode
```bash
python app.py
```

#### Production Mode
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

The API will be available at `http://localhost:5000`

## API Usage Examples

### Teacher Sign-in
```bash
curl -X POST http://localhost:5000/api/auth/signin \
  -H "Content-Type: application/json" \
  -d '{
    "email": "teacher@misrbase.edu",
    "password": "password123"
  }'
```

### Single Prediction
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "QuestionText": "What is 1/2 + 1/4?",
    "MC_Answer": "3/4",
    "StudentExplanation": "I added the numerators and denominators: 1+1=2 and 2+4=6, so 2/6 = 1/3"
  }'
```

### Batch Prediction
```bash
curl -X POST http://localhost:5000/api/batch-predict \
  -H "Content-Type: application/json" \
  -d '{
    "responses": [
      {
        "QuestionText": "What is 1/2 + 1/4?",
        "MC_Answer": "3/4",
        "StudentExplanation": "I added the numerators and denominators"
      }
    ]
  }'
```

## Model Training

To train your own models using the `map-competition.py` script:

1. Prepare your training data in CSV format
2. Run the training script:
   ```bash
   python map-competition.py
   ```
3. Upload the trained models to Hugging Face or save them locally

## Deployment

### Docker Deployment

1. Create a `Dockerfile`:
   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   EXPOSE 5000
   CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
   ```

2. Build and run:
   ```bash
   docker build -t misrbase-backend .
   docker run -p 5000:5000 misrbase-backend
   ```

### Cloud Deployment

The API can be deployed to various cloud platforms:

- **Heroku**: Use the provided `Procfile`
- **AWS**: Deploy using Elastic Beanstalk or ECS
- **Google Cloud**: Use Cloud Run or App Engine
- **Azure**: Use App Service

## Troubleshooting

### Common Issues

1. **Models not loading**: Check that model paths are correct and files exist
2. **CUDA errors**: Set `USE_CUDA=False` in environment variables
3. **CORS errors**: Update `CORS_ORIGINS` to include your frontend URL
4. **Memory issues**: Reduce batch size or use CPU instead of GPU

### Logs

Check the console output for detailed error messages and model loading status.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
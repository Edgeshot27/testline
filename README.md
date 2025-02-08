# NEET Rank Prediction API

This FastAPI-based project predicts a NEET candidate's rank based on quiz scores using a trained Random Forest model.

## Features
- Predicts NEET rank based on quiz performance
- Analyzes quiz results and categorizes users into learning levels (Beginner, Intermediate, Advanced)
- AI-generated learning recommendations using Groq API

## Setup Instructions
### 1. Clone the Repository
```bash
git clone <repo-url>
cd <repo-folder>
```

### 2. Create a Virtual Environment
```bash
python -m venv env
source env/bin/activate  # On Windows use: env\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the project root and add your Groq API key:
```env
GROQ_API_KEY=your_api_key_here
```

### 5. Run the Application
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints
### 1. Analyze Quiz Results
**Endpoint:** `POST /analyze-quiz/`

**Request:**
```json
{
  "user_id": "12345"
}
```

**Response:**
```json
{
  "average_score": 85.6,
  "ai_analysis": "Student is at an Intermediate level. Focus on Physics."
}
```

### 2. Predict NEET Rank
**Endpoint:** `GET /predict-rank/{user_id}`

**Example Request:**
```bash
curl -X GET "http://127.0.0.1:8000/predict-rank/12345"
```

**Example Response:**
```json
{
  "predicted_rank": 1500
}
```

## Dataset Used
- **neet_2024_results.csv**: Contains `Marks` and `Rank` data used to train the model.

## Technologies Used
- FastAPI
- Pandas & NumPy
- Scikit-Learn (Random Forest)
- Joblib (Model Persistence)
- Groq API (AI Recommendations)

## License
This project is licensed under the MIT License.

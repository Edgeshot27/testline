import os
from groq import Groq
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import requests
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

app = FastAPI()

# API Endpoints
QUIZ_ENDPOINT = "https://jsonkeeper.com/b/LLQT"
QUIZ_SUBMISSION_ENDPOINT = "https://api.jsonserve.com/rJvd7g"
HISTORICAL_QUIZ_ENDPOINT = "https://api.jsonserve.com/XgAgFJ"
COLLEGE_DATASET = "neet_2024_results.csv"

# Groq API Setup
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Load NEET 2024 data
neet_data = pd.read_csv(COLLEGE_DATASET)

# Train Rank Prediction Model
def train_rank_predictor(neet_data):
    X = neet_data[['Score']].values
    y = neet_data['Rank'].values
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, "rank_predictor.pkl")
    return model

rank_model = train_rank_predictor(neet_data)

# AI Persona Analysis
def ai_persona_analysis(user_id, user_data, weak_topics, weak_difficulties):
    prompt = f"""
    A student has taken multiple quizzes for NEET preparation. 
    Their past performance is analyzed as follows:
    - Weak Topics: {weak_topics}
    - Weak Difficulty Levels: {weak_difficulties}
    - Past Scores: {[entry['score'] for entry in user_data]}
    
    Based on this, classify the student as "Beginner", "Intermediate", or "Advanced" and suggest improvement strategies.
    """
    chat_completion = client.chat.completions.create(
        messages=[{"role": "system", "content": "You are an expert NEET mentor."},
                  {"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
    )
    return chat_completion.choices[0].message.content

@app.post("/analyze-quiz/")
def analyze_quiz(user_id: str):
    response = requests.get(HISTORICAL_QUIZ_ENDPOINT)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Error fetching quiz data")
    user_data = [entry for entry in response.json() if entry['user_id'] == user_id]
    if not user_data:
        raise HTTPException(status_code=404, detail="User data not found")
    avg_score = np.mean([entry['score'] for entry in user_data])
    ai_analysis = ai_persona_analysis(user_id, user_data, [], [])
    return {"average_score": avg_score, "ai_analysis": ai_analysis}

@app.get("/predict-rank/{user_id}")
def predict_rank(user_id: str):
    response = requests.get(HISTORICAL_QUIZ_ENDPOINT)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Error fetching historical quiz data")
    user_data = [entry for entry in response.json() if entry['user_id'] == user_id]
    if not user_data:
        raise HTTPException(status_code=404, detail="User data not found")
    predicted_rank = rank_model.predict([[np.mean([entry['score'] for entry in user_data])]])[0]
    return {"predicted_rank": int(predicted_rank)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

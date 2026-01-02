from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

logistic_model = joblib.load("logistic_model.joblib")
tree_model = joblib.load("decision_tree_model.joblib")

class InputData(BaseModel):
    features: list

@app.post("/predict/logistic")
def predict_logistic(data: InputData):
    pred = logistic_model.predict([data.features])
    return {"prediction": int(pred[0])}

@app.post("/predict/tree")
def predict_tree(data: InputData):
    pred = tree_model.predict([data.features])
    return {"prediction": int(pred[0])}

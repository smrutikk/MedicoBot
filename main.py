from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Load your trained model
model = pickle.load(open('model.pkl', 'rb'))

# Define the Pydantic model for input validation
class Symptoms(BaseModel):
    symptom_1: str
    symptom_2: str
    symptom_3: str
    symptom_4: str
    symptom_5: str
    symptom_6: str

# Define the symptom mapping based on your model training
symptoms_map = {
    'itching': 0,
    'skin_rash': 1,
    'nodal_skin_eruptions': 2,
    # Add all symptoms with their corresponding encodings used during training
    # Example:
    'dischromic_patches': 3,
    # Continue this mapping for all symptoms
}

@app.post("/predict")
def predict(symptoms: Symptoms):
    try:
        symptoms_list = [
            symptoms_map[symptoms.symptom_1],
            symptoms_map[symptoms.symptom_2],
            symptoms_map[symptoms.symptom_3],
            symptoms_map[symptoms.symptom_4],
            symptoms_map[symptoms.symptom_5],
            symptoms_map[symptoms.symptom_6]
        ]
        symptoms_array = np.array([symptoms_list])
        prediction = model.predict(symptoms_array)
        return {"prediction": prediction[0]}
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Symptom {str(e)} not recognized")

# For testing the deployment with Dialogflow
@app.post("/dialogflow-webhook")
async def dialogflow_webhook(request: Request):
    req = await request.json()
    symptoms = req['queryResult']['parameters']
    symptoms_list = [
        symptoms['symptom_1'],
        symptoms['symptom_2'],
        symptoms['symptom_3'],
        symptoms['symptom_4'],
        symptoms['symptom_5'],
        symptoms['symptom_6']
    ]
    
    prediction_response = requests.post('http://EXTERNAL_IP/predict', json={
        "symptom_1": symptoms_list[0],
        "symptom_2": symptoms_list[1],
        "symptom_3": symptoms_list[2],
        "symptom_4": symptoms_list[3],
        "symptom_5": symptoms_list[4],
        "symptom_6": symptoms_list[5]
    })
    
    prediction = prediction_response.json().get('prediction')
    
    response = {
        "fulfillmentText": f"Based on the symptoms you described, you might be experiencing {prediction}. However, I recommend consulting a healthcare professional for an accurate diagnosis."
    }
    
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


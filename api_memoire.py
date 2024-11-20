from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json
import numpy as np


app=FastAPI(


title="AVC_KILLER",
    description=""" ** Introduction to the Stroke Prediction API **

Welcome to the documentation for the Stroke Prediction API, a powerful tool designed to help healthcare professionals, researchers and developers assess an individual's risk of developing a stroke. This API, based on FastAPI technology and powered by advanced machine learning models, accurately predicts the likelihood of a person developing a stroke based on various risk factors.

** API objective **

The main objective of our API is to provide a user-friendly interface for stroke risk assessment, using data such as age, body mass index, glucose levels and smoking. Thanks to this API, users can obtain risk predictions quickly and efficiently, which can be essential for stroke prevention and management.


** Using the API **

To use our Stroke Prediction API, you need to submit a patient dataset as an HTTP POST request to the `/avc_prediction` endpoint. The API will then return a response containing the stroke risk prediction for the data provided.


** Documentation structure **


This documentation will guide you through the API's features, show you how to make requests, interpret responses, and provide you with code examples for practical use. We hope you will find this API invaluable in your efforts to prevent and manage stroke.

We invite you to explore the following sections to fully understand how the Stroke Prediction API works, and to start integrating it into your applications and projects.


** Endpoints: **


Endpoint 1 : /avc_prediction


Méthode HTTP : POST


Description : This endpoint allows users to submit data for stroke prediction.


** Example query: **

{

  "avg_level_glucose": 228.9 (cg/L),
  "bmi": 33.7 (kg/m2),
  "age": 55,
  "smoking_status": "formerly smoked",
}



** Example of response: **

{
  "prediction": "Not Safe"
}



** Possible status codes ** :


200 OK : The prediction was successful.


400 Bad Request: The request is incorrect or missing.


401 Unauthorized: The user is not authenticated (if applicable).


500 Internal Server Error: An internal error has occurred.

""",
    summary="FIGHT AVC",
    version="0.0.1",
    
)

class model_input(BaseModel):
    Glucose : float
    BMI : float
    Smoking : str
    Age : float

        
#loading model
avc_model=pickle.load( open(r'C:\Users\Ninette HOUKPONOU\Repertoire_python\Memoire\notebooks_data_without_sleep_Time\avc_model.sav', 'rb') )

avc_scaler=pickle.load( open(r'C:\Users\Ninette HOUKPONOU\Repertoire_python\Memoire\notebooks_data_without_sleep_Time\avc_scaler.pkl', 'rb') )


@app.post('/avc_prediction')
def avc_pred(input_parameters : model_input):
    input_data=input_parameters.json()
    input_dictionary = json.loads(input_data)
    

    avg_glucose_level = input_dictionary['Glucose']
    bmi = input_dictionary['BMI']
    smoking_status = input_dictionary['Smoking']
    age = input_dictionary['Age']
    
    
    if smoking_status == "formerly smoked":
        smoking_status=1
    elif smoking_status== "never smoked":
        smoking_status=2
    elif smoking_status== "smokes":
        smoking_status=3
    else :
        smoking_status=4
    
    
    
    scaled_data=[[age,avg_glucose_level,bmi]]
    scaled_data=avc_scaler.transform(scaled_data)
    
    for i in range (2):
        age=scaled_data[0,0]
        avg_glucose_level=scaled_data[0,1]
        bmi=scaled_data[0,2]
    
    input_data_as_list=[avg_glucose_level,bmi,age,smoking_status]
    input_data_as_numpy=np.asarray(input_data_as_list)
    input_data_reshaped=input_data_as_numpy.reshape(1,-1)

   
    prediction=avc_model.predict(input_data_reshaped)
    
    
    if prediction[0]==0 :
        return "Safe"
    
    else :
        return "Not Safe"  
    
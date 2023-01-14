"""
Created on Dec 2022
@author: Yasmine UNEAU
"""

# 1. Library imports

from fastapi import FastAPI, HTTPException,  Request
import pickle
import pandas as pd
import uvicorn
import shap
import json
import io
import matplotlib.pyplot as plt
from starlette.responses import Response

# 2. Data import
X = pd.read_csv("df_sample.csv",index_col= ['SK_ID_CURR'], encoding ='utf-8')


# 3. Create the app object
app = FastAPI(title="Default payment risk prediction API", description="API for default payment risk prediction using Lightgbm machine learning model")

@app.get('/')
def get_root():
    return {'message': 'Welcome to the default payment risk prediction API'}

# 4. Loading the trained model
pickle_in = open("LightGBMmodel.pkl","rb")
classifier=pickle.load(pickle_in)

# 5. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted payment risk

""" 
Get customer id from test file 
"""
@app.get("/customer/{sk_id_cust}")
async def get_customer_id(sk_id_cust:int):
    X_cust = X.loc[sk_id_cust: sk_id_cust]
    return {"id_client": X_cust}

""" 
Get default payment risk prediction from test file 
"""
@app.get("/prediction")
def get_prediction(sk_id_cust: int):     
    # get customer id using their id as row index
    X_cust = X.loc[sk_id_cust: sk_id_cust]
    
    # get prediction , O for no risk and 1 for risk 
    score_cust = classifier.predict(X_cust)[0]

    # Return the prediction as a JSON dictionary
    if(not(sk_id_cust)):
        raise HTTPException(status_code=400, 
                            detail = "Please Provide a valid data")

    if (score_cust == 0):
        prediction_explanation = "No payment default risk : LOAN GRANTED"
    else:
        prediction_explanation = "Payment default risk : LOAN REJECTED" 
    
    return {
        'prediction': score_cust,
        'prediction_explanation': prediction_explanation
    }


if __name__ == '__main__':
    uvicorn.run(app)



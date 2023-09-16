import numpy as np
from catboost import CatBoostClassifier

def get_encoded_value(input_val, col,encodings):
  """
  Get the encoded value for the given input data
  """
  return encodings[col][input_val] 
    


def get_prediction(data,model):
    """
    Predict the class of a given data point.
    """
    result= model.predict(data)
    if result[0]==0:
      return "Slight Injury"
    elif result[0]==1:
      return "Serious Injury"
    else:
      return "Fatal Injury"

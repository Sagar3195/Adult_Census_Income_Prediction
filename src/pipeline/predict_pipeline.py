import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd 
import numpy as np
from dataclasses import dataclass

from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass


    def predict(self,features):
        try:
            ## Load pickel File
            preprocessor_path = os.path.join("artifacts","preprocessor.pkl")
            model_path = os.path.join("artifacts","model.pkl")

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)

            pred = model.predict(data_scaled)

            return pred

        except Exception as e:
            logging.info("Error Occure in Prediction Pipline")
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
            age:int,
            workclass:int,
            education_num:int,
            marital_status:int,
            occupation:int,
            relationship:int,
            race:int,
            sex:int,
            capital_gain:int,
            capital_loss:int,
            hours_per_week:int,
            country:int):

        self.age = age
        self.workclass = workclass
        self.education_num = education_num
        self.marital_status = marital_status
        self.occupation = occupation
        self.relationship = relationship
        self.race = race
        self.sex = sex
        self.capital_gain = capital_gain
        self.capital_loss = capital_loss
        self.hours_per_week = hours_per_week
        self.country = country
        

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "age":[self.age],
                "workclass":[self.workclass], 
                "education_num":[self.education_num], 
                "marital_status":[self.marital_status],
                "occupation":[self.occupation],
                "relationship":[self.relationship],
                "race":[self.race], 
                "sex":[self.sex],
                "capital_gain":[self.capital_gain], 
                "capital_loss":[self.capital_loss], 
                "hours_per_week":[self.hours_per_week], 
                "country":[self.country] 
            }

            data = pd.DataFrame(custom_data_input_dict)
            logging.info("Data Frame Gathered")
            return data

        except Exception as e:
            logging.info("Error Occured In Predict Pipline")
            raise CustomException(e, sys)



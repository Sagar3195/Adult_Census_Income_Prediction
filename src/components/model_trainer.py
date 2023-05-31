import os 
import sys
import pickle
import numpy as np
import pandas as pd
from src.exception import CustomException 
from src.logger import logging 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as ltb
from src.utils import save_object, evaluate_model
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    train_model_file_path= os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config= ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Splitting Indepenedent an dependent variable from train and test dataset.")
            X_train, y_train, X_test, y_test= (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]

            )

            #Train the multiple models
            models = {
                "Logistic Regression": LogisticRegression(),
                "K-Neighbors Classifier": KNeighborsClassifier(),
                "Support Vector Machine": SVC(),
                "Random Forest Classifier": RandomForestClassifier(),
                "Decision Tree Classifier": DecisionTreeClassifier(), 
                "AdaBoost Classifier": AdaBoostClassifier(),
                "XGBoost Classifier": xgb.XGBClassifier(),
                "LGBM Classifier": ltb.LGBMClassifier()
            }
            
            model_report:dict= evaluate_model(X_train, y_train, X_test, y_test,models)
            print(model_report)
            print("\n=\n"*5)
            logging.info(f"Model Report: {model_report}")

            #To get best model score from dictionary 
            best_model_score= max(sorted(model_report.values()))

            best_model_name= list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model= models[best_model_name]

            print(f"Best Model Found, Model Name: {best_model_name}, Accuracy Score: {best_model_score}")
            print("\n=\n"*5)
            logging.info(f"Best Model Found, Model Name: {best_model_name}, Accuracy Score: {best_model_score}")

            save_object(
                file_path= self.model_trainer_config.train_model_file_path,
                obj= best_model
            )



        except Exception as e:
            logging.info("Exception occured at Model Training")
            raise CustomException(e, sys)



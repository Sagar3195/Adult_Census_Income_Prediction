import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config= DataTransformationConfig()



    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            numerical_columns= ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
                                'hours-per-week']
            categorical_columns= ['workclass', 'education', 'marital-status', 'occupation',
                                  'relationship', 'race', 'sex', 'country']

            num_pipeline= Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='median')),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline= Pipeline(
                steps= [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown= 'ignore',sparse_output=False)),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical Columns: {categorical_columns}")
            logging.info(f"Numerical Columns: {numerical_columns}")


            preprocessor= ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )


            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed.")
            logging.info("Obtaining preprocessing object.")

            preprocessing_obj= self.get_data_transformer_object()
            
            #Target variable
            target_column_name = "salary"
            
            #Here we will drop education and fnlwgt columns from dataset
            #Education column has already encoded as education num column in dataset

            drop_columns= ['education', 'fnlwgt', target_column_name]

            #Encoding Target variable 
            train_df['salary'] = train_df['salary'].map({'<=50K':0, '>50K':1})            
            test_df['salary'] = test_df['salary'].map({'<=50K':0, '>50K':1})
            
            #In workclass, occupation and country features has '?' value 
            #So we will replace this value with  np.nan values.
            train_df['workclass'] = train_df['workclass'].replace("?", np.nan)
            train_df['occupation'] = train_df['occupation'].replace("?", np.nan)
            train_df['country'] = train_df['country'].replace("?", np.nan)

            #Similarly for test df, we will replace '?' value with np.nan value
            test_df['workclass'] = test_df['workclass'].replace("?", np.nan)
            test_df['occupation'] = test_df['occupation'].replace("?", np.nan)
            test_df['country'] = test_df['country'].replace("?", np.nan)

            input_feature_train_df= train_df.drop(columns= [drop_columns], axis=1)
            target_feature_train_df= train_df[target_column_name]

            input_feature_test_df= test_df.drop(columns= [drop_columns], axis=1)
            target_feature_test_df= test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr= preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr= preprocessing_obj.transform(input_feature_test_df)

            train_arr= np.c_[input_feature_train_arr, np.array(target_feature_train_df)]

            test_arr= np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path= self.data_transformation_config.preprocessor_obj_file_path,
                obj= preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,

            )


        except Exception as e:
            raise CustomException(e, sys)


            
import os
import sys
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

if __name__=='__main__':
    obj= DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    print(train_data_path, test_data_path)
    data_transformation= DataTransformation()
    data_transformation.initiate_data_transformation(train_data_path, test_data_path)

    

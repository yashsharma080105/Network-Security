import yaml
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
import os,sys
import numpy as np
# import dill
import pickle
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV


def read_yaml_file(file_path:str)->dict:
    try:
        with open(file_path,"rb") as yaml_file:
            return yaml.safe_load(yaml_file)
        
    except Exception as e:
        raise NetworkSecurityException(e,sys) from e
    

def write_yaml_file(file_path: str,content: object,replace: bool=False)->None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"w") as file:
            yaml.dump(content,file)
    except Exception as e:
        raise NetworkSecurityException(e,sys)
    
def save_numpy_array_data(file_path: str,array:np.array):

    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            np.save(file_obj,array)

    except Exception as e:
        raise NetworkSecurityException(e,sys) from e
    

def save_object(file_path:str,obj:object) -> None:
    try:
        logging.info("Entered the save_obj method of Mainutils class")
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
        logging.info("excited the save_obj method of mainutils class")
    except Exception as e:
        raise NetworkSecurityException(e,sys) from e
    
    
def load_object(file_path: str, )->object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path,"rb") as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e,sys) from e
    
def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return : np.array data loaded
    """

    try:
        with open(file_path,"rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e,sys) from e


def evaluate_models(X_train, y_train, X_test, y_test, models: dict, params: dict) -> dict:
    """Train and evaluate a collection of models.

    For each entry in ``models`` the corresponding parameter grid from
    ``params`` is used to perform a GridSearchCV. If no parameter grid is
    provided the estimator is trained as‑is.

    The dictionary passed in ``models`` is mutated so that each key's value
    becomes the fitted estimator (either the result of grid search or the
    original model). The returned report maps model names to their F1 score
    on the test set.
    """
    try:
        report: dict = {}
        for model_name, model in models.items():
            model_params = params.get(model_name, {})
            if model_params:
                grid = GridSearchCV(
                    estimator=model,
                    param_grid=model_params,
                    scoring="f1",
                    cv=3,
                    n_jobs=-1,
                )
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_
            else:
                model.fit(X_train, y_train)
                best_model = model

            # update model reference so caller can use tuned version later
            models[model_name] = best_model

            y_pred = best_model.predict(X_test)
            score = f1_score(y_test, y_pred)
            report[model_name] = score
        return report
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
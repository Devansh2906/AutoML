import os
import pandas as pd
from pycaret.classification import ClassificationExperiment
from pycaret.regression import RegressionExperiment

class AutoMLBackend:
    def __init__(self):
        self.exp = None
        self.problem_type = None
        self.best_model = None

    def detect_problem_type(self, df: pd.DataFrame, target: str) -> str:
        unique_values = df[target].nunique()
        data_type = df[target].dtype

        # If data type is text/object, or if numerical values are very few, treat as classification
        if data_type == 'object' or data_type == 'bool' or unique_values < 10:
            return 'classification'
        else:
            return 'regression'

    def run_pipeline(self, df: pd.DataFrame, target: str):
        # Step 1: Detect problem type
        self.problem_type = self.detect_problem_type(df, target)
        
        # Step 2: Initialize correct model
        if self.problem_type == 'classification':
            self.exp = ClassificationExperiment()
        else:
            self.exp = RegressionExperiment()

        # Step 3: Setup Data for PyCaret
        self.exp.setup(
            data=df, 
            target=target, 
            session_id=42, 
            verbose=True, # Prints progress in your background terminal window!
            profile=False
        )

        # Step 4: Trains and Compares models only in Syllabus
        if self.problem_type == 'classification':
            # Logistic Regression ('lr'), Decision Trees ('dt'), Random Forests ('rf'), SVMs ('svm')
            self.best_model = self.exp.compare_models(
                include=['lr', 'dt', 'rf', 'svm'], 
                sort='F1', 
                fold=3
            )
        else:
            # Linear Regression ('lr'), Decision Trees ('dt'), Random Forests ('rf'), Support Vector Regression ('svm')
            self.best_model = self.exp.compare_models(
                include=['lr', 'dt', 'rf', 'svm'], 
                sort='RMSE', 
                fold=3
            )
        
        # Step 5: Extract the leaderboard metrics dataframe
        leaderboard = self.exp.pull()
        
        return self.problem_type, leaderboard

    def save_winning_model(self, model_dir="saved_models", filename="best_pipeline"):
        """
        Serializes the entire pipeline (processing + model weights) to disk.
        """
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        path = os.path.join(model_dir, filename)
        self.exp.save_model(self.best_model, path)
        return f"{path}.pkl"
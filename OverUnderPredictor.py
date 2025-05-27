import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib


class OverUnderPredictor:
    def __init__(self, model_name='RandomForestClassifier', model_config=None, model_path='over_under_model.pkl'):
        self.model_name = model_name
        self.model_config = model_config or {}
        self.model_path = model_path
        self.model = None

    def build_model(self):
        if self.model_name == 'RandomForestClassifier':
            clf = RandomForestClassifier(**self.model_config)
        elif self.model_name == 'LogisticRegression':
            clf = LogisticRegression(**self.model_config)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        pipeline = make_pipeline(
            SimpleImputer(strategy='mean'),
            StandardScaler(),
            clf
        )
        self.model = pipeline

    def train(self, df: pd.DataFrame, target_col='Over25', save_model=True):
        if 'Over25' not in df.columns:
            raise ValueError("Target column 'Over25' not found in training data")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.build_model()
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Over/Under 2.5 Model Accuracy: {acc:.3f}")

        if save_model:
            joblib.dump(self.model, self.model_path)
            print(f"Model saved to {self.model_path}")

    def load(self):
        self.model = joblib.load(self.model_path)
        print(f"Model loaded from {self.model_path}")

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Model is not trained or loaded.")
        return pd.Series(self.model.predict(df), index=df.index, name='Over25_Pred')

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model is not trained or loaded.")
        proba = self.model.predict_proba(df)
        return pd.DataFrame(proba, columns=['Under2.5_Prob', 'Over2.5_Prob'], index=df.index)
import os
import json
import warnings
from typing import Union, Tuple

import pandas as pd
import numpy as np
from sklearn import preprocessing, compose
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_selector

import game

def add_ckfm_columns(dataset: pd.DataFrame, model_proba: np.ndarray, odds_cols=['B365H', 'B365D', 'B365A']) -> pd.DataFrame:
    odds = dataset[odds_cols].values
    market_probs = 1 / odds
    market_probs = market_probs / market_probs.sum(axis=1, keepdims=True)
    ckfm = model_proba - market_probs
    dataset = dataset.copy()
    dataset[['ckfm_H', 'ckfm_D', 'ckfm_A']] = ckfm
    return dataset

def dataset_preprocessing(
    dataset: pd.DataFrame,
    label_encoder: Union[None, preprocessing.LabelEncoder] = None,
    feature_preprocessor: Union[None, compose.ColumnTransformer] = None
) -> Tuple[np.ndarray, np.ndarray, preprocessing.LabelEncoder, compose.ColumnTransformer]:
    Y = dataset['result']
    X = dataset.drop(columns=['result'])
    if label_encoder is None:
        label_encoder = preprocessing.LabelEncoder().fit(Y)
    Y_encoded = label_encoder.transform(Y)
    if feature_preprocessor is None:
        feature_preprocessor = compose.ColumnTransformer(transformers=[
            ('cat', preprocessing.OneHotEncoder(handle_unknown='ignore'), make_column_selector(dtype_include=object)),
            ('num', make_pipeline(SimpleImputer(strategy='mean'), preprocessing.StandardScaler()),
             make_column_selector(dtype_exclude=object))
        ])
        feature_preprocessor.fit(X)
    X_transformed = feature_preprocessor.transform(X)
    return X_transformed, Y_encoded, label_encoder, feature_preprocessor

class HomeTeamWins:
    """始终预测主队赢的 baseline 策略"""
    def __init__(self):
        self.label_encoder = None
    def infer(self, dataset: pd.DataFrame, **kwargs) -> pd.DataFrame:
        pred = pd.Series(['H'] * len(dataset), index=dataset.index)
        encoded = pd.Series(self.label_encoder.transform(pred), index=dataset.index)
        return pd.concat([pred, encoded], axis=1, keys=['result', 'encoded_result'])

class CKFMValueBetting:
    """
    CKFM baseline: 选出ckfm_H/D/A中分数最大的那个作为预测
    """
    def __init__(self):
        self.label_encoder = None
    def infer(self, dataset: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ckfm_cols = ['ckfm_H', 'ckfm_D', 'ckfm_A']
        if not all(col in dataset.columns for col in ckfm_cols):
            raise ValueError(f"数据中未找到CKFM相关列：{ckfm_cols}")
        max_idxs = dataset[ckfm_cols].values.argmax(axis=1)
        label_map = {0: 'H', 1: 'D', 2: 'A'}
        pred = pd.Series([label_map[i] for i in max_idxs], index=dataset.index)
        encoded = pd.Series(self.label_encoder.transform(pred), index=dataset.index)
        return pd.concat([pred, encoded], axis=1, keys=['result', 'encoded_result'])

class ResultsPredictor:
    def __init__(self, league: 'game.League', **kwargs):
        self.league = league
        self.model_name = kwargs['model_name']
        config_path = kwargs.get('config_name') or os.path.join('configs', f'{self.model_name}.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.model = eval(self.model_name)(**config)
        print(f"\nModel for predicting the game results: {self.model.__class__.__name__}")
        self.training_dataset, self.test_dataset = self._split_train_test()
        self.label_encoder = None
        self.feature_preprocessor = None
        self.baselines = [HomeTeamWins(), CKFMValueBetting()]
        self.predicted_matches = {}

    def _split_train_test(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        training = [self.league.datasets[season.name] for season in self.league.seasons[:-1]]
        training_df = pd.concat(training)
        test_df = self.league.datasets[self.league.seasons[-1].name]
        if len(self.league.seasons) == 1:
            warnings.warn("Only one season provided. Training and test set are the same.")
        return training_df, test_df

    def train(self):
        X, Y, self.label_encoder, self.feature_preprocessor = dataset_preprocessing(self.training_dataset)
        print(f"Feature set: {list(self.training_dataset.drop(columns='result').columns)}")
        print(f"Feature set size: {X.shape[1]}")
        print(f"\nTraining on {len(self.training_dataset)} matches from {self.league.seasons[0].name} "
              f"to {self.league.seasons[-2].name}...")
        self.model.fit(X, Y)
        train_pred = self.model.predict(X)
        print(f"Training accuracy: {accuracy_score(Y, train_pred):.3f}")

    def eval(self, with_heuristics=True):
        print(f"\nEvaluating on {len(self.test_dataset)} matches from season {self.league.seasons[-1].name}...")
        X, Y, _, _ = dataset_preprocessing(self.test_dataset, self.label_encoder, self.feature_preprocessor)
        pred = self.model.predict(X)
        print(f"Model accuracy: {accuracy_score(Y, pred):.3f}")

        if with_heuristics:
            print("="*60)
            print("Baseline evaluations:")
            print("="*60)
            for baseline in self.baselines:
                baseline.label_encoder = self.label_encoder
                if isinstance(baseline, CKFMValueBetting):
                    # 生成带有ckfm列的数据（必须）
                    test_data_with_ckfm = self.infer(self.test_dataset, with_proba=True, with_ckfm=True)
                    try:
                        baseline_pred = baseline.infer(test_data_with_ckfm)
                        acc = accuracy_score(Y, baseline_pred['encoded_result'])
                        print(f"{baseline.__class__.__name__} accuracy: {acc:.3f}")
                    except Exception as e:
                        print(f"⚠️ {baseline.__class__.__name__} baseline无法评估: {e}")
                else:
                    try:
                        baseline_pred = baseline.infer(self.test_dataset)
                        acc = accuracy_score(Y, baseline_pred['encoded_result'])
                        print(f"{baseline.__class__.__name__} accuracy: {acc:.3f}")
                    except Exception as e:
                        print(f"⚠️ {baseline.__class__.__name__} baseline无法评估: {e}")

    def infer(self, dataset: pd.DataFrame, with_proba=False, with_ckfm=False, season_name=None) -> pd.DataFrame:
        X, _, _, _ = dataset_preprocessing(dataset, self.label_encoder, self.feature_preprocessor)
        if isinstance(X, pd.DataFrame):
            X = X.fillna(X.mean())
        elif isinstance(X, np.ndarray) and np.isnan(X).any():
            imputer = SimpleImputer(strategy='mean')
            X = imputer.fit_transform(X)
        # 主体预测
        if with_proba:
            proba_df = pd.DataFrame(self.model.predict_proba(X),
                                    columns=self.label_encoder.classes_,
                                    index=dataset.index)
            proba_df['result'] = proba_df.idxmax(axis=1)
            result_df = proba_df
        else:
            pred = self.model.predict(X)
            labels = self.label_encoder.inverse_transform(pred)
            result_df = pd.DataFrame({'result': labels}, index=dataset.index)
        # 动态生成CKFM列（只要with_ckfm就主动生成）
        if with_ckfm:
            odds_cols = ['B365H', 'B365D', 'B365A']
            if all(col in dataset.columns for col in odds_cols):
                odds = dataset[odds_cols].values
                market_probs = 1 / odds
                market_probs = market_probs / market_probs.sum(axis=1, keepdims=True)
                proba_ckfm = self.model.predict_proba(X)
                ckfm = proba_ckfm - market_probs
                dataset = dataset.copy()
                dataset[['ckfm_H', 'ckfm_D', 'ckfm_A']] = ckfm
                result_df = result_df.join(dataset[['ckfm_H', 'ckfm_D', 'ckfm_A']])
            else:
                print("⚠️ CKFM无法生成：数据中缺少赔率字段 B365H/D/A")
        if season_name:
            self.predicted_matches[season_name] = pd.concat([dataset.reset_index(drop=True),
                                                             result_df.reset_index(drop=True)], axis=1)
        return result_df
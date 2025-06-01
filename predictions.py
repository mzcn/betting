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


def add_ckfm_columns(dataset: pd.DataFrame, model_proba: np.ndarray,
                     odds_cols=['B365H', 'B365D', 'B365A']) -> pd.DataFrame:
    """添加CKFM列到数据集"""
    odds = dataset[odds_cols].values
    market_probs = 1 / odds
    market_probs = market_probs / market_probs.sum(axis=1, keepdims=True)
    ckfm = model_proba - market_probs
    dataset = dataset.copy()
    dataset[['ckfm_H', 'ckfm_D', 'ckfm_A']] = ckfm
    return dataset


def calculate_ckfm_from_odds_and_proba(odds_data: pd.DataFrame, model_proba: np.ndarray,
                                       odds_cols=['B365H', 'B365D', 'B365A']) -> pd.DataFrame:
    """从赔率数据和模型概率计算CKFM"""
    if not all(col in odds_data.columns for col in odds_cols):
        raise ValueError(f"缺少赔率列：{odds_cols}")

    odds = odds_data[odds_cols].values
    # 计算市场隐含概率
    market_probs = 1 / odds
    market_probs = market_probs / market_probs.sum(axis=1, keepdims=True)

    # 计算CKFM
    ckfm = model_proba - market_probs

    result_df = pd.DataFrame({
        'ckfm_H': ckfm[:, 0],
        'ckfm_D': ckfm[:, 1],
        'ckfm_A': ckfm[:, 2]
    }, index=odds_data.index)

    return result_df


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
    注意：这个类不再作为独立预测器使用，而是作为CKFM计算的工具类
    """

    def __init__(self):
        self.label_encoder = None

    def infer_from_ckfm(self, ckfm_data: pd.DataFrame) -> pd.DataFrame:
        """从已计算的CKFM数据进行推断"""
        ckfm_cols = ['ckfm_H', 'ckfm_D', 'ckfm_A']
        if not all(col in ckfm_data.columns for col in ckfm_cols):
            raise ValueError(f"数据中未找到CKFM相关列：{ckfm_cols}")

        max_idxs = ckfm_data[ckfm_cols].values.argmax(axis=1)
        label_map = {0: 'H', 1: 'D', 2: 'A'}
        pred = pd.Series([label_map[i] for i in max_idxs], index=ckfm_data.index)
        encoded = pd.Series(self.label_encoder.transform(pred), index=ckfm_data.index)
        return pd.concat([pred, encoded], axis=1, keys=['result', 'encoded_result'])

    def infer(self, dataset: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """保持向后兼容的接口"""
        return self.infer_from_ckfm(dataset)


class ResultsPredictor:
    def __init__(self, league: 'game.League', **kwargs):
        self.league = league
        self.model_name = kwargs['model_name']
        config_path = kwargs.get('config_name') or os.path.join('configs', f'{self.model_name}.json')

        # 如果配置文件不存在，使用默认配置
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            print(f"配置文件 {config_path} 不存在，使用默认配置")
            config = self._get_default_config()

        self.model = eval(self.model_name)(**config)
        print(f"\nModel for predicting the game results: {self.model.__class__.__name__}")

        self.training_dataset, self.test_dataset = self._split_train_test()
        self.label_encoder = None
        self.feature_preprocessor = None
        self.baselines = [HomeTeamWins()]  # 移除CKFMValueBetting，稍后单独处理
        self.predicted_matches = {}

    def _get_default_config(self) -> dict:
        """获取默认模型配置"""
        default_configs = {
            'LogisticRegression': {'random_state': 42, 'max_iter': 1000},
            'RandomForestClassifier': {'n_estimators': 100, 'random_state': 42},
            'MLPClassifier': {'hidden_layer_sizes': (100,), 'random_state': 42, 'max_iter': 500},
            'DecisionTreeClassifier': {'random_state': 42}
        }
        return default_configs.get(self.model_name, {})

    def _split_train_test(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        training = [self.league.datasets[season.name] for season in self.league.seasons[:-1]]
        training_df = pd.concat(training) if training else pd.DataFrame()
        test_df = self.league.datasets[self.league.seasons[-1].name]
        if len(self.league.seasons) == 1:
            warnings.warn("Only one season provided. Training and test set are the same.")
            training_df = test_df.copy()
        return training_df, test_df

    def train(self):
        if self.training_dataset.empty:
            print("⚠️ 训练集为空，跳过训练")
            return

        X, Y, self.label_encoder, self.feature_preprocessor = dataset_preprocessing(self.training_dataset)
        print(f"Feature set: {list(self.training_dataset.drop(columns='result').columns)}")
        print(f"Feature set size: {X.shape[1]}")
        print(f"\nTraining on {len(self.training_dataset)} matches from {self.league.seasons[0].name} "
              f"to {self.league.seasons[-2].name if len(self.league.seasons) > 1 else self.league.seasons[-1].name}...")

        self.model.fit(X, Y)
        train_pred = self.model.predict(X)
        print(f"Training accuracy: {accuracy_score(Y, train_pred):.3f}")

    def eval(self, with_heuristics=True):
        if self.test_dataset.empty:
            print("⚠️ 测试集为空，跳过评估")
            return

        print(f"\nEvaluating on {len(self.test_dataset)} matches from season {self.league.seasons[-1].name}...")
        X, Y, _, _ = dataset_preprocessing(self.test_dataset, self.label_encoder, self.feature_preprocessor)
        pred = self.model.predict(X)
        print(f"Model accuracy: {accuracy_score(Y, pred):.3f}")

        if with_heuristics:
            print("=" * 60)
            print("Baseline evaluations:")
            print("=" * 60)

            # 评估基础baseline
            for baseline in self.baselines:
                baseline.label_encoder = self.label_encoder
                try:
                    baseline_pred = baseline.infer(self.test_dataset)
                    acc = accuracy_score(Y, baseline_pred['encoded_result'])
                    print(f"{baseline.__class__.__name__} accuracy: {acc:.3f}")
                except Exception as e:
                    print(f"⚠️ {baseline.__class__.__name__} baseline无法评估: {e}")

            # 评估CKFM策略
            self._eval_ckfm_strategy(Y)

    def _eval_ckfm_strategy(self, Y):
        """单独评估CKFM策略"""
        try:
            # 首先获取模型预测概率
            test_data_with_proba = self.infer(self.test_dataset, with_proba=True)

            # 寻找可用的赔率列
            available_odds_cols = self._find_available_odds_columns()

            if available_odds_cols:
                print(f"使用赔率列: {available_odds_cols}")
                model_proba = test_data_with_proba[['H', 'D', 'A']].values
                ckfm_data = calculate_ckfm_from_odds_and_proba(self.test_dataset, model_proba, available_odds_cols)

                # 使用CKFM进行预测
                ckfm_baseline = CKFMValueBetting()
                ckfm_baseline.label_encoder = self.label_encoder
                ckfm_pred = ckfm_baseline.infer_from_ckfm(ckfm_data)
                acc = accuracy_score(Y, ckfm_pred['encoded_result'])
                print(f"CKFMValueBetting accuracy: {acc:.3f}")
            else:
                print("⚠️ CKFMValueBetting baseline无法评估: 缺少赔率数据")
                print("可用的列:", list(self.test_dataset.columns))
                self._try_generate_mock_odds()
        except Exception as e:
            print(f"⚠️ CKFMValueBetting baseline无法评估: {e}")

    def _find_available_odds_columns(self):
        """寻找可用的赔率列"""
        betting_platforms = ['B365', 'BW', 'IW', 'PS', 'WH', 'VC']
        odds_suffixes = ['H', 'D', 'A']

        for platform in betting_platforms:
            odds_cols = [f"{platform}{suffix}" for suffix in odds_suffixes]
            if all(col in self.test_dataset.columns for col in odds_cols):
                # 检查是否有足够的非空数据
                non_null_counts = [self.test_dataset[col].notna().sum() for col in odds_cols]
                if all(count > len(self.test_dataset) * 0.1 for count in non_null_counts):  # 至少10%的数据
                    return odds_cols

        return None

    def _try_generate_mock_odds(self):
        """尝试生成模拟赔率数据进行CKFM演示"""
        print("🔧 生成模拟赔率数据用于CKFM演示...")
        try:
            from data_inspector import patch_dataset_with_mock_odds

            # 为测试数据添加模拟赔率
            test_with_odds = patch_dataset_with_mock_odds(self.test_dataset, 'B365')

            # 重新计算CKFM
            test_data_with_proba = self.infer(self.test_dataset, with_proba=True)
            model_proba = test_data_with_proba[['H', 'D', 'A']].values
            odds_cols = ['B365H', 'B365D', 'B365A']

            ckfm_data = calculate_ckfm_from_odds_and_proba(test_with_odds, model_proba, odds_cols)

            # 使用CKFM进行预测
            ckfm_baseline = CKFMValueBetting()
            ckfm_baseline.label_encoder = self.label_encoder
            ckfm_pred = ckfm_baseline.infer_from_ckfm(ckfm_data)

            # 计算准确率
            X, Y, _, _ = dataset_preprocessing(self.test_dataset, self.label_encoder, self.feature_preprocessor)
            acc = accuracy_score(Y, ckfm_pred['encoded_result'])
            print(f"CKFMValueBetting (模拟赔率) accuracy: {acc:.3f}")

        except Exception as e:
            print(f"⚠️ 模拟赔率生成失败: {e}")

    def infer(self, dataset: pd.DataFrame, with_proba=False, with_ckfm=False, season_name=None) -> pd.DataFrame:
        X, _, _, _ = dataset_preprocessing(dataset, self.label_encoder, self.feature_preprocessor)

        # 处理缺失值
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

        # 动态生成CKFM列
        if with_ckfm:
            odds_cols = ['B365H', 'B365D', 'B365A']
            if all(col in dataset.columns for col in odds_cols):
                try:
                    proba_for_ckfm = self.model.predict_proba(X)
                    ckfm_data = calculate_ckfm_from_odds_and_proba(dataset, proba_for_ckfm, odds_cols)
                    result_df = result_df.join(ckfm_data)
                except Exception as e:
                    print(f"⚠️ CKFM计算失败: {e}")
            else:
                print("⚠️ CKFM无法生成：数据中缺少赔率字段 B365H/D/A")

        if season_name:
            self.predicted_matches[season_name] = pd.concat([dataset.reset_index(drop=True),
                                                             result_df.reset_index(drop=True)], axis=1)
        return result_df

    def get_ckfm_predictions(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """专门用于获取CKFM预测的方法"""
        # 先获取模型概率预测
        proba_result = self.infer(dataset, with_proba=True)

        # 计算CKFM
        odds_cols = ['B365H', 'B365D', 'B365A']
        if not all(col in dataset.columns for col in odds_cols):
            raise ValueError(f"缺少赔率列用于CKFM计算: {odds_cols}")

        model_proba = proba_result[['H', 'D', 'A']].values
        ckfm_data = calculate_ckfm_from_odds_and_proba(dataset, model_proba, odds_cols)

        # 使用CKFM进行预测
        ckfm_baseline = CKFMValueBetting()
        ckfm_baseline.label_encoder = self.label_encoder
        ckfm_pred = ckfm_baseline.infer_from_ckfm(ckfm_data)

        return ckfm_pred
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.impute import SimpleImputer
import predictions

class BettingStrategy:
    def __init__(self, results_predictor: 'predictions.ResultsPredictor', evaluate_heuristics=True, **kwargs):
        self.results_predictor = results_predictor
        self.initial_bankroll = kwargs['initial_bankroll']
        self.betting_platform = kwargs['betting_platform']
        self.stake_per_bet = kwargs['stake_per_bet']
        self.do_value_betting = kwargs['do_value_betting']
        self.value_betting_on_all_results = kwargs['value_betting_on_all_results']
        self.analyze_betting_platforms_margins = kwargs.get('analyze_betting_platforms_margins', False)
        self.evaluate_ckfm = kwargs.get("evaluate_ckfm", False)

        print(f"\nInitial bankroll: {self.initial_bankroll:.2f}")
        print(f"Bet platform: {self.betting_platform}")
        print(f"Stake per bet: {self.stake_per_bet:.2f}")
        print(f"Bet only on EV+ results: {self.do_value_betting}")

        self.all_predictors = {self.results_predictor.model_name: self.results_predictor}
        if evaluate_heuristics:
            for baseline in self.results_predictor.baselines:
                self.all_predictors[baseline.__class__.__name__] = baseline
        if self.evaluate_ckfm:
            # 加入CKFM baseline策略（只要你有定义 predictions.CKFMValueBetting）
            ckfm = predictions.CKFMValueBetting()
            self.all_predictors['CKFMValueBetting'] = ckfm

        self.total_bet_amount = {name: 0 for name in self.all_predictors}
        self.bankroll = {name: self.initial_bankroll for name in self.all_predictors}
        self.bankroll_over_time = {}
        self.ckfm_records = []

    def apply(self, dataset: pd.DataFrame, matches: pd.DataFrame, verbose=False):
        for name, predictor in self.all_predictors.items():
            current_data = dataset.copy()
            with_proba = (name == self.results_predictor.model_name) and self.do_value_betting

            # 只在主模型做缺失值填补
            if with_proba and current_data.isna().any().any():
                print("\u26a0\ufe0f NaNs detected. Using SimpleImputer to fill numeric columns.")
                numeric_cols = current_data.select_dtypes(include=[np.number]).columns
                valid_cols = [col for col in numeric_cols if current_data[col].notna().any()]
                if valid_cols:
                    imputer = SimpleImputer(strategy='mean')
                    current_data[valid_cols] = imputer.fit_transform(current_data[valid_cols])

            # 各predictor分别推断
            try:
                if name == 'CKFMValueBetting':
                    predictor.label_encoder = self.results_predictor.label_encoder
                    predictions = predictor.infer(current_data)
                else:
                    predictions = predictor.infer(current_data, with_proba=with_proba)
            except Exception as e:
                print(f"⚠️ Predictor {name} failed: {e}")
                continue

            for idx, match in matches.iterrows():
                if idx not in predictions.index:
                    if verbose:
                        print(f"\u26a0\ufe0f Missing prediction: {match.get('HomeTeam')} vs {match.get('AwayTeam')}")
                    continue

                bet_result = self._select_bet_result(predictions, idx, match, with_proba)
                if bet_result is None or self.bankroll[name] < self.stake_per_bet:
                    continue

                self.total_bet_amount[name] += self.stake_per_bet
                self.bankroll[name] -= self.stake_per_bet

                if match.get('FTR') == bet_result:
                    odds = match.get(f"{self.betting_platform}{match['FTR']}", np.nan)
                    if pd.notna(odds) and odds > 1:
                        self.bankroll[name] += self.stake_per_bet * odds

    def _select_bet_result(self, predictions: pd.DataFrame, idx, match, with_proba):
        if not with_proba:
            return predictions.loc[idx, 'result']

        best_value = 0
        selected_result = None
        outcomes = ['H', 'D', 'A'] if self.value_betting_on_all_results else [predictions.loc[idx, 'result']]

        total_market_inv = sum(1 / match.get(f"{self.betting_platform}{x}", np.nan) for x in outcomes if match.get(f"{self.betting_platform}{x}", 0) > 1)
        if total_market_inv == 0:
            return None

        for outcome in outcomes:
            odds = match.get(f"{self.betting_platform}{outcome}", np.nan)
            if pd.isna(odds) or odds <= 1:
                continue

            market_prob = 1 / odds / total_market_inv
            model_prob = predictions.loc[idx, outcome]
            ckfm = model_prob - market_prob
            expected_value = model_prob * odds

            if ckfm > 0.05 and expected_value > 1 and expected_value > best_value:
                selected_result = outcome
                best_value = expected_value
                if self.evaluate_ckfm:
                    self.ckfm_records.append(ckfm)
                    print(f"[CKFM] \u2705 Bet: {outcome} | CKFM: {ckfm:.3f} | EV: {expected_value:.3f}")

        return selected_result

    def record_bankroll(self, date: pd.Timestamp):
        self.bankroll_over_time[date] = {
            name: self.bankroll[name] for name in self.all_predictors
        }

    def display_results(self):
        sns.set()
        df = pd.DataFrame(self.bankroll_over_time)
        for name in self.all_predictors:
            print(f"Predictor: {name}")
            print(f"   Total bet: ${self.total_bet_amount[name]:.2f}")
            print(f"   Final bankroll: ${self.bankroll[name]:.2f}\n")
            df.loc[name].plot(label=name)

        plt.xlabel("Date")
        plt.ylabel("Bankroll")
        plt.title(f"Betting Performance ({self.betting_platform}, Stake = {self.stake_per_bet})")
        plt.legend()
        plt.tight_layout()
        plt.show()

        if self.ckfm_records:
            plt.figure()
            sns.histplot(self.ckfm_records, bins=30, kde=True)
            plt.title("CKFM Score Distribution")
            plt.xlabel("CKFM = Model - Market Probability")
            plt.tight_layout()
            plt.show()
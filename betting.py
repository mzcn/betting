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

        # 初始化预测器字典
        self.all_predictors = {self.results_predictor.model_name: self.results_predictor}

        # 添加基础baseline策略
        if evaluate_heuristics:
            for baseline in self.results_predictor.baselines:
                baseline.label_encoder = self.results_predictor.label_encoder
                self.all_predictors[baseline.__class__.__name__] = baseline

        # 如果启用CKFM评估，添加特殊标记
        if self.evaluate_ckfm:
            self.all_predictors['CKFMValueBetting'] = 'special_ckfm'  # 特殊标记

        self.total_bet_amount = {name: 0 for name in self.all_predictors}
        self.bankroll = {name: self.initial_bankroll for name in self.all_predictors}
        self.bankroll_over_time = {}
        self.ckfm_records = []

    def apply(self, dataset: pd.DataFrame, matches: pd.DataFrame, verbose=False):
        """应用投注策略"""
        for name, predictor in self.all_predictors.items():
            if name == 'CKFMValueBetting' and predictor == 'special_ckfm':
                # 特殊处理CKFM策略
                self._apply_ckfm_strategy(dataset, matches, verbose)
                continue

            current_data = dataset.copy()
            with_proba = (name == self.results_predictor.model_name) and self.do_value_betting

            # 只在主模型做缺失值填补
            if with_proba and current_data.isna().any().any():
                if verbose:
                    print("⚠️ NaNs detected. Using SimpleImputer to fill numeric columns.")
                numeric_cols = current_data.select_dtypes(include=[np.number]).columns
                valid_cols = [col for col in numeric_cols if current_data[col].notna().any()]
                if valid_cols:
                    imputer = SimpleImputer(strategy='mean')
                    current_data[valid_cols] = imputer.fit_transform(current_data[valid_cols])

            # 各predictor分别推断
            try:
                predictions_result = predictor.infer(current_data, with_proba=with_proba)
            except Exception as e:
                if verbose:
                    print(f"⚠️ Predictor {name} failed: {e}")
                continue

            # 应用投注逻辑
            self._apply_betting_logic(name, predictions_result, matches, with_proba, verbose)

    def _apply_ckfm_strategy(self, dataset: pd.DataFrame, matches: pd.DataFrame, verbose=False):
        """专门处理CKFM投注策略"""
        name = 'CKFMValueBetting'
        try:
            # 使用results_predictor的专门方法获取CKFM预测
            ckfm_predictions = self.results_predictor.get_ckfm_predictions(dataset)
            self._apply_betting_logic(name, ckfm_predictions, matches, with_proba=False, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"⚠️ CKFM策略失败: {e}")

    def _apply_betting_logic(self, predictor_name: str, predictions: pd.DataFrame,
                             matches: pd.DataFrame, with_proba: bool, verbose=False):
        """应用具体的投注逻辑"""
        for idx, match in matches.iterrows():
            if idx not in predictions.index:
                if verbose:
                    print(
                        f"⚠️ Missing prediction for {predictor_name}: {match.get('HomeTeam')} vs {match.get('AwayTeam')}")
                continue

            bet_result = self._select_bet_result(predictions, idx, match, with_proba, predictor_name)
            if bet_result is None or self.bankroll[predictor_name] < self.stake_per_bet:
                continue

            self.total_bet_amount[predictor_name] += self.stake_per_bet
            self.bankroll[predictor_name] -= self.stake_per_bet

            # 检查投注结果
            actual_result = match.get('FTR')
            if actual_result == bet_result:
                odds_key = f"{self.betting_platform}{actual_result}"
                odds = match.get(odds_key, np.nan)
                if pd.notna(odds) and odds > 1:
                    winnings = self.stake_per_bet * odds
                    self.bankroll[predictor_name] += winnings
                    if verbose and predictor_name == 'CKFMValueBetting':
                        print(f"✅ CKFM Win: {bet_result} @ {odds:.2f} -> +${winnings:.2f}")

    def _select_bet_result(self, predictions: pd.DataFrame, idx, match, with_proba, predictor_name=None):
        """选择投注结果"""
        if not with_proba:
            return predictions.loc[idx, 'result']

        # 价值投注逻辑
        best_value = 0
        selected_result = None

        if self.value_betting_on_all_results:
            outcomes = ['H', 'D', 'A']
        else:
            # 只考虑模型预测的最可能结果
            outcomes = [predictions.loc[idx, 'result']]

        # 计算市场总概率用于归一化
        total_market_inv = 0
        for outcome in ['H', 'D', 'A']:
            odds = match.get(f"{self.betting_platform}{outcome}", np.nan)
            if pd.notna(odds) and odds > 1:
                total_market_inv += 1 / odds

        if total_market_inv == 0:
            return None

        for outcome in outcomes:
            odds_key = f"{self.betting_platform}{outcome}"
            odds = match.get(odds_key, np.nan)
            if pd.isna(odds) or odds <= 1:
                continue

            market_prob = (1 / odds) / total_market_inv
            model_prob = predictions.loc[idx, outcome]
            ckfm = model_prob - market_prob
            expected_value = model_prob * odds

            # 设置投注阈值
            ckfm_threshold = 0.05
            ev_threshold = 1.0

            if self.do_value_betting:
                # 价值投注：要求正期望值和足够的CKFM优势
                if ckfm > ckfm_threshold and expected_value > ev_threshold and expected_value > best_value:
                    selected_result = outcome
                    best_value = expected_value

                    # 记录CKFM信息
                    if self.evaluate_ckfm and predictor_name == self.results_predictor.model_name:
                        self.ckfm_records.append(ckfm)
                        print(
                            f"[CKFM] ✅ Bet: {outcome} | CKFM: {ckfm:.3f} | EV: {expected_value:.3f} | Odds: {odds:.2f}")
            else:
                # 非价值投注：直接选择最可能的结果
                if expected_value > best_value:
                    selected_result = outcome
                    best_value = expected_value

        return selected_result

    def record_bankroll(self, date: pd.Timestamp):
        """记录资金变化"""
        self.bankroll_over_time[date] = {
            name: self.bankroll[name] for name in self.all_predictors
        }

    def display_results(self):
        """显示投注结果"""
        # 设置绘图样式
        plt.style.use('default')
        sns.set_palette("husl")

        # 打印结果摘要
        print("\n" + "=" * 50)
        print("投注策略结果摘要:")
        print("=" * 50)

        for name in self.all_predictors:
            profit = self.bankroll[name] - self.initial_bankroll
            roi = (profit / self.initial_bankroll) * 100 if self.initial_bankroll > 0 else 0

            print(f"\n📊 {name}:")
            print(f"   初始资金: ${self.initial_bankroll:.2f}")
            print(f"   总投注金额: ${self.total_bet_amount[name]:.2f}")
            print(f"   最终资金: ${self.bankroll[name]:.2f}")
            print(f"   盈亏: ${profit:.2f}")
            print(f"   投资回报率: {roi:.2f}%")

            if self.total_bet_amount[name] > 0:
                bet_roi = (profit / self.total_bet_amount[name]) * 100
                print(f"   投注回报率: {bet_roi:.2f}%")

        # 绘制资金变化图
        if self.bankroll_over_time:
            plt.figure(figsize=(12, 6))
            df = pd.DataFrame(self.bankroll_over_time).T

            for name in self.all_predictors:
                if name in df.columns:
                    plt.plot(df.index, df[name], label=name, linewidth=2)

            plt.axhline(y=self.initial_bankroll, color='gray', linestyle='--', alpha=0.7, label='Initial Bankroll')
            plt.xlabel("Date", fontsize=12)
            plt.ylabel("Bankroll ($)", fontsize=12)
            plt.title(f"Betting Performance Comparison\n({self.betting_platform}, Stake=${self.stake_per_bet})",
                      fontsize=14)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        # 绘制CKFM分布图
        if self.ckfm_records:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.ckfm_records, bins=30, kde=True, alpha=0.7)
            plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Zero CKFM')
            plt.xlabel("CKFM Score (Model Prob - Market Prob)", fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            plt.title("CKFM Score Distribution for Value Bets", fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

            # CKFM统计摘要
            ckfm_array = np.array(self.ckfm_records)
            print(f"\n📈 CKFM统计摘要:")
            print(f"   总CKFM投注次数: {len(self.ckfm_records)}")
            print(f"   平均CKFM: {ckfm_array.mean():.4f}")
            print(f"   CKFM标准差: {ckfm_array.std():.4f}")
            print(f"   最大CKFM: {ckfm_array.max():.4f}")
            print(f"   最小CKFM: {ckfm_array.min():.4f}")

    def get_performance_summary(self) -> dict:
        """获取性能摘要数据"""
        summary = {}
        for name in self.all_predictors:
            profit = self.bankroll[name] - self.initial_bankroll
            roi = (profit / self.initial_bankroll) * 100 if self.initial_bankroll > 0 else 0

            summary[name] = {
                'initial_bankroll': self.initial_bankroll,
                'final_bankroll': self.bankroll[name],
                'total_bet_amount': self.total_bet_amount[name],
                'profit': profit,
                'roi': roi,
                'num_bets': self.total_bet_amount[name] / self.stake_per_bet if self.stake_per_bet > 0 else 0
            }

        return summary
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

        # 如果启用CKFM评估，添加CKFM策略
        if self.evaluate_ckfm:
            # 创建CKFM策略实例
            ckfm_strategy = predictions.CKFMValueBetting()
            ckfm_strategy.label_encoder = self.results_predictor.label_encoder
            self.all_predictors['CKFMValueBetting'] = ckfm_strategy

        self.total_bet_amount = {name: 0 for name in self.all_predictors}
        self.bankroll = {name: self.initial_bankroll for name in self.all_predictors}
        self.bankroll_over_time = {}
        self.ckfm_records = []
        self.bet_records = {name: [] for name in self.all_predictors}  # 记录每个策略的投注

    def apply(self, dataset: pd.DataFrame, matches: pd.DataFrame, verbose=False):
        """应用投注策略"""
        for name, predictor in self.all_predictors.items():
            current_data = dataset.copy()

            # 对于CKFM策略，需要特殊处理
            if name == 'CKFMValueBetting':
                self._apply_ckfm_strategy(current_data, matches, verbose)
                continue

            # 其他策略的正常处理
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
            # 获取主模型的概率预测
            main_predictions = self.results_predictor.infer(dataset, with_proba=True)

            # 检查赔率数据
            odds_cols = [f"{self.betting_platform}{suffix}" for suffix in ['H', 'D', 'A']]

            # 确保matches和dataset有相同的索引
            for idx in dataset.index:
                if idx not in matches.index:
                    continue

                match = matches.loc[idx]

                # 检查赔率是否可用
                if not all(col in match.index and pd.notna(match[col]) and match[col] > 1 for col in odds_cols):
                    continue

                # 如果没有足够资金，跳过
                if self.bankroll[name] < self.stake_per_bet:
                    continue

                # 获取模型概率
                model_probs = main_predictions.loc[idx, ['H', 'D', 'A']].values

                # 计算市场概率
                odds = [match[col] for col in odds_cols]
                market_probs = np.array([1 / o for o in odds])
                market_probs = market_probs / market_probs.sum()

                # 计算CKFM
                ckfm_scores = model_probs - market_probs

                # 找出CKFM最高的选项
                best_idx = np.argmax(ckfm_scores)
                best_outcome = ['H', 'D', 'A'][best_idx]
                best_ckfm = ckfm_scores[best_idx]
                best_odds = odds[best_idx]

                # CKFM策略：只在CKFM为正且超过阈值时下注
                ckfm_threshold = 0.05  # 5%的优势
                if best_ckfm > ckfm_threshold:
                    # 进行投注
                    self.total_bet_amount[name] += self.stake_per_bet
                    self.bankroll[name] -= self.stake_per_bet

                    # 检查投注结果
                    actual_result = match.get('FTR')
                    if actual_result == best_outcome:
                        winnings = self.stake_per_bet * best_odds
                        self.bankroll[name] += winnings
                        profit = winnings - self.stake_per_bet
                        if verbose:
                            print(
                                f"[CKFM] ✅ Win: {best_outcome} @ {best_odds:.2f} -> +${profit:.2f} (CKFM: {best_ckfm:.3f})")
                    else:
                        if verbose:
                            print(f"[CKFM] ❌ Loss: Bet {best_outcome}, Result {actual_result} (CKFM: {best_ckfm:.3f})")

                    # 记录投注信息
                    self.bet_records[name].append({
                        'match_idx': idx,
                        'bet': best_outcome,
                        'odds': best_odds,
                        'ckfm': best_ckfm,
                        'result': actual_result,
                        'won': actual_result == best_outcome
                    })

                    # 记录CKFM分数
                    self.ckfm_records.append(best_ckfm)

        except Exception as e:
            if verbose:
                print(f"⚠️ CKFM策略失败: {e}")
                import traceback
                traceback.print_exc()

    def _apply_betting_logic(self, predictor_name: str, predictions: pd.DataFrame,
                             matches: pd.DataFrame, with_proba: bool, verbose=False):
        """应用具体的投注逻辑"""
        for idx, match in matches.iterrows():
            if idx not in predictions.index:
                if verbose:
                    print(
                        f"⚠️ Missing prediction for {predictor_name}: {match.get('HomeTeam')} vs {match.get('AwayTeam')}")
                continue

            bet_result = self._select_bet_result(predictions, idx, match, with_proba, predictor_name, verbose)
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

            # 记录投注
            self.bet_records[predictor_name].append({
                'match_idx': idx,
                'bet': bet_result,
                'result': actual_result,
                'won': actual_result == bet_result
            })

    def _select_bet_result(self, predictions: pd.DataFrame, idx, match, with_proba, predictor_name=None, verbose=False):
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

        for outcome in outcomes:
            odds_key = f"{self.betting_platform}{outcome}"
            odds = match.get(odds_key, np.nan)
            if pd.isna(odds) or odds <= 1:
                continue

            model_prob = predictions.loc[idx, outcome]
            expected_value = model_prob * odds

            # 设置投注阈值
            ev_threshold = 1.0

            if self.do_value_betting:
                # 价值投注：要求正期望值
                if expected_value > ev_threshold and expected_value > best_value:
                    selected_result = outcome
                    best_value = expected_value

                    if verbose and predictor_name == self.results_predictor.model_name:
                        print(f"[{predictor_name}] Value bet: {outcome} | EV: {expected_value:.3f} | Odds: {odds:.2f}")
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
            num_bets = len(self.bet_records[name])
            wins = sum(1 for bet in self.bet_records[name] if bet['won'])
            win_rate = (wins / num_bets * 100) if num_bets > 0 else 0

            print(f"\n📊 {name}:")
            print(f"   初始资金: ${self.initial_bankroll:.2f}")
            print(f"   总投注金额: ${self.total_bet_amount[name]:.2f}")
            print(f"   最终资金: ${self.bankroll[name]:.2f}")
            print(f"   盈亏: ${profit:.2f}")
            print(f"   投资回报率: {roi:.2f}%")
            print(f"   投注次数: {num_bets}")
            print(f"   胜率: {win_rate:.1f}% ({wins}/{num_bets})")

            if self.total_bet_amount[name] > 0:
                bet_roi = (profit / self.total_bet_amount[name]) * 100
                print(f"   投注回报率: {bet_roi:.2f}%")

        # 绘制资金变化图
        if self.bankroll_over_time:
            plt.figure(figsize=(12, 6))
            df = pd.DataFrame(self.bankroll_over_time).T

            for name in self.all_predictors:
                if name in df.columns:
                    plt.plot(df.index, df[name], label=name, linewidth=2, marker='o', markersize=4)

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
            plt.hist(self.ckfm_records, bins=30, alpha=0.7, color='blue', edgecolor='black')
            plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Zero CKFM')
            plt.axvline(x=0.05, color='green', linestyle='--', alpha=0.7, label='Threshold (5%)')
            plt.xlabel("CKFM Score (Model Prob - Market Prob)", fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            plt.title("CKFM Score Distribution for Bets", fontsize=14)
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
            num_bets = len(self.bet_records[name])
            wins = sum(1 for bet in self.bet_records[name] if bet['won'])
            win_rate = (wins / num_bets * 100) if num_bets > 0 else 0

            summary[name] = {
                'initial_bankroll': self.initial_bankroll,
                'final_bankroll': self.bankroll[name],
                'total_bet_amount': self.total_bet_amount[name],
                'profit': profit,
                'roi': roi,
                'num_bets': num_bets,
                'wins': wins,
                'win_rate': win_rate
            }

        return summary
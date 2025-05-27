import pandas as pd
import numpy as np
from typing import Optional, List

def generate_mock_ckfm_sample(
    n_matches: int = 1000,
    output_csv: str = 'ckfm_training_sample.csv',
    odds_ranges: Optional[dict] = None,
    random_state: int = 42
):
    np.random.seed(random_state)
    if odds_ranges is None:
        odds_ranges = {'H': (1.4, 3.5), 'D': (2.8, 4.5), 'A': (1.6, 4.2)}

    proba = np.random.dirichlet(alpha=[1, 1, 1], size=n_matches)
    df = pd.DataFrame(proba, columns=['proba_H', 'proba_D', 'proba_A'])

    # 统一用真实流程的赔率字段名
    for key, col in zip(['H', 'D', 'A'], ['B365H', 'B365D', 'B365A']):
        low, high = odds_ranges[key]
        df[col] = np.round(np.random.uniform(low, high, size=n_matches), 2)

    inv_odds = 1 / df[['B365H', 'B365D', 'B365A']]
    market_probs = inv_odds.div(inv_odds.sum(axis=1), axis=0)
    df[['market_prob_H', 'market_prob_D', 'market_prob_A']] = market_probs

    df['ckfm_H'] = df['proba_H'] - df['market_prob_H']
    df['ckfm_D'] = df['proba_D'] - df['market_prob_D']
    df['ckfm_A'] = df['proba_A'] - df['market_prob_A']

    df['FTR'] = np.random.choice(['H', 'D', 'A'], size=n_matches, p=[0.45, 0.25, 0.30])

    df.to_csv(output_csv, index=False)
    print(f"✅ 模拟样本文件已生成：{output_csv}")

def collect_real_ckfm_from_predictor(
    results_predictor,
    output_csv: str = 'ckfm_training_sample.csv',
    odds_columns: Optional[List[str]] = None
):
    """
    odds_columns 默认为 ['B365H', 'B365D', 'B365A']，即Bet365的欧赔
    """
    if odds_columns is None:
        odds_columns = ['B365H', 'B365D', 'B365A']

    ckfm_data = []
    for season in results_predictor.league.seasons:
        matches = results_predictor.league.datasets[season.name]
        pred_proba = results_predictor.infer(matches, with_proba=True)
        for idx, row in matches.iterrows():
            # 检查赔率和FTR都齐全
            if row.isnull().any():
                continue
            try:
                odds = [row.get(col, np.nan) for col in odds_columns]
                if any(pd.isna(o) or o <= 1 for o in odds):
                    continue
                inv_odds = [1/o for o in odds]
                total_inv = sum(inv_odds)
                if total_inv == 0:
                    continue
                market_probs = [p / total_inv for p in inv_odds]
                # 模型概率
                if isinstance(pred_proba, pd.DataFrame):
                    model_probs = pred_proba.loc[idx, ['H', 'D', 'A']].tolist()
                else:
                    model_probs = list(pred_proba[idx])
                ckfm = [m - b for m, b in zip(model_probs, market_probs)]

                ckfm_data.append({
                    'HomeRanking': row.get('HomeRanking', np.nan),
                    'AwayRanking': row.get('AwayRanking', np.nan),
                    odds_columns[0]: odds[0],
                    odds_columns[1]: odds[1],
                    odds_columns[2]: odds[2],
                    'proba_H': model_probs[0],
                    'proba_D': model_probs[1],
                    'proba_A': model_probs[2],
                    'market_H': market_probs[0],
                    'market_D': market_probs[1],
                    'market_A': market_probs[2],
                    'ckfm_H': ckfm[0],
                    'ckfm_D': ckfm[1],
                    'ckfm_A': ckfm[2],
                    'FTR': row['FTR']
                })
            except Exception as e:
                print(f"⚠️ Error processing match index {idx}: {e}")
                continue

    df_ckfm = pd.DataFrame(ckfm_data)
    df_ckfm.to_csv(output_csv, index=False)
    print(f"✅ 实际历史样本已生成：{output_csv}")

# ========== 示例使用 ==========
if __name__ == '__main__':
    # 1. 生成模拟样本
    generate_mock_ckfm_sample(n_matches=1000, output_csv='ckfm_training_sample.csv')

    # 2. 生成真实数据
    # from main import results_predictor
    # collect_real_ckfm_from_predictor(results_predictor, odds_columns=['B365H', 'B365D', 'B365A'])
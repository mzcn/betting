"""
Enhanced Main file for the Sports Bet project with data inspection
"""
import argparse
import pandas as pd
import numpy as np
from typing import Dict, Union, List
import os

from game import League
from predictions import ResultsPredictor
from betting import BettingStrategy

pd.options.display.width = 0
np.seterr(divide='ignore', invalid='ignore')
betting_platforms = ['B365', 'BW', 'IW', 'PS', 'WH', 'VC']


def inspect_league_data(league: League):
    """检查联赛数据质量"""
    print("\n🔍 数据质量检查")
    print("=" * 50)

    for i, season in enumerate(league.seasons):
        print(f"\n📅 赛季 {season.name}:")
        print(f"   比赛数量: {len(season.matches)}")

        if not season.matches.empty:
            # 检查基本列
            basic_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTR', 'FTHG', 'FTAG']
            missing_basic = [col for col in basic_cols if col not in season.matches.columns]
            if missing_basic:
                print(f"   ⚠️ 缺少基本列: {missing_basic}")
            else:
                print(f"   ✅ 基本列完整")

            # 检查赔率列
            odds_platforms_available = []
            for platform in betting_platforms:
                odds_cols = [f"{platform}{suffix}" for suffix in ['H', 'D', 'A']]
                if all(col in season.matches.columns for col in odds_cols):
                    non_null_counts = [season.matches[col].notna().sum() for col in odds_cols]
                    if all(count > 0 for count in non_null_counts):
                        odds_platforms_available.append(platform)
                        print(f"   ✅ {platform} 赔率: {non_null_counts}")

            if not odds_platforms_available:
                print(f"   ❌ 无可用赔率数据")
                print(f"   可用列: {list(season.matches.columns)}")

            # 显示数据样本
            print(f"   📊 数据样本:")
            if len(season.matches) > 0:
                sample = season.matches.head(2)
                for idx, row in sample.iterrows():
                    print(
                        f"      {row.get('HomeTeam', 'N/A')} vs {row.get('AwayTeam', 'N/A')} - {row.get('FTR', 'N/A')}")


def patch_league_with_odds(league: League, platform: str = 'B365'):
    """为联赛数据添加模拟赔率"""
    print(f"\n🔧 为联赛数据添加模拟 {platform} 赔率...")

    for season in league.seasons:
        if season.matches.empty:
            continue

        odds_cols = [f"{platform}{suffix}" for suffix in ['H', 'D', 'A']]

        # 检查是否已经有完整赔率
        if all(col in season.matches.columns for col in odds_cols):
            non_null_counts = [season.matches[col].notna().sum() for col in odds_cols]
            if all(count > len(season.matches) * 0.8 for count in non_null_counts):
                print(f"   ✅ 赛季 {season.name} 已有足够的 {platform} 赔率数据")
                continue

        print(f"   🔧 为赛季 {season.name} 生成模拟赔率...")

        # 生成模拟赔率
        np.random.seed(hash(season.name) % 2 ** 32)  # 使用赛季名作为种子保证一致性
        n_matches = len(season.matches)

        for i in range(n_matches):
            # 基于历史数据的典型赔率分布
            base_probs = np.random.dirichlet([2.2, 1.0, 1.8])  # 主队略有优势
            margin = np.random.uniform(1.08, 1.12)  # 博彩公司边际

            odds = margin / base_probs
            odds = np.clip(odds, 1.1, 15.0)  # 合理范围

            season.matches.loc[season.matches.index[i], f'{platform}H'] = round(odds[0], 2)
            season.matches.loc[season.matches.index[i], f'{platform}D'] = round(odds[1], 2)
            season.matches.loc[season.matches.index[i], f'{platform}A'] = round(odds[2], 2)

        print(f"   ✅ 已为 {n_matches} 场比赛添加模拟赔率")


def ensure_configs_directory():
    """确保configs目录存在并创建默认配置文件"""
    configs_dir = 'configs'
    if not os.path.exists(configs_dir):
        os.makedirs(configs_dir)
        print(f"✅ 创建配置目录: {configs_dir}")

    # 创建默认配置文件
    default_configs = {
        'LogisticRegression.json': {
            "random_state": 42,
            "max_iter": 1000,
            "solver": "lbfgs"
        },
        'RandomForestClassifier.json': {
            "n_estimators": 100,
            "random_state": 42,
            "max_depth": 10,
            "min_samples_split": 5
        },
        'MLPClassifier.json': {
            "hidden_layer_sizes": [100, 50],
            "random_state": 42,
            "max_iter": 500,
            "alpha": 0.001
        },
        'DecisionTreeClassifier.json': {
            "random_state": 42,
            "max_depth": 10,
            "min_samples_split": 10
        }
    }

    import json
    for filename, config in default_configs.items():
        config_path = os.path.join(configs_dir, filename)
        if not os.path.exists(config_path):
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"✅ 创建默认配置文件: {config_path}")


def parse_arguments(args: Union[None, List] = None) -> Dict:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Sports Betting ML Prediction System")

    # Game
    parser.add_argument('--country', required=True, type=str,
                        help="Name of the league's country, among {Italy, France, Spain, England, Germany}.")
    parser.add_argument('--division', type=int, default=1,
                        help="Division level, e.g. 1 for Premier League in England. By default, set to 1.")
    parser.add_argument('--start_season', required=True, type=str,
                        help="Two digit year indicating the first season analyzed, e.g. 04 for the 2004/2005 season.")
    parser.add_argument('--end_season', required=False,
                        help="Two digit year indicating the last season analyzed.")
    parser.add_argument('--evaluate_ckfm', action='store_true',
                        help='Evaluate using CKFM and print recommended bets')

    # Betting
    parser.add_argument('--betting_platform', required=False, type=str, default="B365",
                        help="Ticker of the betting platform, among {B365, BW, IW, PS, WH, VC}.")
    parser.add_argument('--initial_bankroll', type=float, default=100,
                        help="Initial amount allowed for betting. By default, set to 100.")
    parser.add_argument('--stake_per_bet', type=float, default=1,
                        help="Stake for each bet. By default, set to 1.")
    parser.add_argument('--do_value_betting', action="store_true", default=False,
                        help="If true, bet only if the expected value of a result is positive.")
    parser.add_argument('--value_betting_on_all_results', action="store_true", default=False,
                        help="If true, perform value betting on the three results.")
    parser.add_argument('--analyze_betting_platforms_margins', action="store_true", default=False,
                        help="If true, compute the average margins of the betting platforms.")

    # Features
    parser.add_argument('--match_history_length', default=None, type=int,
                        help="Number of previous matches for each facing team included in the model features.")
    parser.add_argument('--number_previous_direct_confrontations', default=3, type=int,
                        help="Use the last k direct confrontations between the two teams as features.")
    parser.add_argument('--match_results_encoding', default='points', type=str,
                        help="Encode match results as 'categorical' or 'points'.")

    # Model
    parser.add_argument('--model_name', default='LogisticRegression', type=str,
                        help="Chosen predictive model: {LogisticRegression, MLPClassifier, DecisionTreeClassifier, RandomForestClassifier}.")
    parser.add_argument('--config_name', default=None, type=str,
                        help="Model configuration name or path.")

    # Options
    parser.add_argument('--verbose', action='store_true', help="Enable verbose output for debugging")
    parser.add_argument('--quick_test', action='store_true', help="Run with minimal data for quick testing")
    parser.add_argument('--generate_mock_odds', action='store_true', help="Generate mock odds data if missing")
    parser.add_argument('--inspect_data', action='store_true', help="Inspect data quality before processing")

    args = parser.parse_args(args)

    # 验证参数
    countries = ['Italy', 'France', 'Spain', 'England', 'Germany']
    assert args.country in countries, f'{args.country} not in supported countries {countries}'

    assert args.betting_platform in betting_platforms, f'{args.betting_platform} not in supported platforms {betting_platforms}'

    models = ['LogisticRegression', 'MLPClassifier', 'DecisionTreeClassifier', 'RandomForestClassifier']
    assert args.model_name in models, f'{args.model_name} not in supported models {models}'

    if not args.end_season:
        args.end_season = f'{(int(args.start_season) + 1) % 100:02d}'

    return vars(args)


def validate_environment():
    """验证运行环境"""
    required_dirs = ['data']
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"✅ 创建目录: {dir_name}")


def main():
    """主函数"""
    print("🏈 体育博彩机器学习预测系统启动")
    print("=" * 50)

    try:
        kwargs = parse_arguments()
        validate_environment()
        ensure_configs_directory()

        print(f"\n📋 运行配置:")
        for key, value in kwargs.items():
            print(f"   {key}: {value}")

        # 初始化联赛
        print(f"\n🏆 初始化联赛数据...")
        league = League(betting_platforms, **kwargs)

        # 数据质量检查
        if kwargs.get('inspect_data', True):  # 默认开启检查
            inspect_league_data(league)

        # 如果需要生成模拟赔率
        if kwargs.get('generate_mock_odds', False) or kwargs.get('evaluate_ckfm', False):
            patch_league_with_odds(league, kwargs['betting_platform'])

        # 运行联赛模拟
        league.run()

        # 分析博彩平台边际（如果请求）
        if kwargs['analyze_betting_platforms_margins']:
            print(f"\n📊 分析博彩平台边际...")
            league.analyze_betting_platforms_margins()

        # 初始化结果预测器
        print(f"\n🤖 初始化预测模型...")
        results_predictor = ResultsPredictor(league, **kwargs)

        # 训练模型
        print(f"\n📚 训练模型...")
        results_predictor.train()

        # 评估模型
        print(f"\n🎯 评估模型...")
        results_predictor.eval()

        # 应用投注策略
        print(f"\n💰 应用投注策略...")
        betting_strategy = BettingStrategy(results_predictor, **kwargs)
        league.seasons[-1].run(betting_strategy)

        # 显示结果
        print(f"\n📈 显示投注结果...")
        betting_strategy.display_results()

        print(f"\n✅ 程序执行完成!")

    except KeyboardInterrupt:
        print(f"\n❌ 程序被用户中断")
    except Exception as e:
        print(f"\n💥 程序执行出错: {e}")
        if kwargs.get('verbose'):
            import traceback
            traceback.print_exc()
        raise


def quick_test():
    """快速测试函数"""
    print("🧪 运行快速测试...")
    test_args = [
        '--country', 'England',
        '--start_season', '19',
        '--end_season', '19',
        '--model_name', 'LogisticRegression',
        '--initial_bankroll', '100',
        '--stake_per_bet', '1',
        '--do_value_betting',
        '--evaluate_ckfm',
        '--generate_mock_odds',  # 自动生成模拟赔率
        '--verbose'
    ]

    kwargs = parse_arguments(test_args)
    validate_environment()
    ensure_configs_directory()

    try:
        print("📋 测试配置:")
        for key, value in kwargs.items():
            print(f"   {key}: {value}")

        league = League(betting_platforms, **kwargs)

        # 检查数据
        inspect_league_data(league)

        # 添加模拟赔率
        patch_league_with_odds(league, kwargs['betting_platform'])

        # 运行
        league.run()

        results_predictor = ResultsPredictor(league, **kwargs)
        results_predictor.train()
        results_predictor.eval()

        # 简化的投注测试
        if len(league.seasons) > 0:
            betting_strategy = BettingStrategy(results_predictor, **kwargs)
            league.seasons[-1].run(betting_strategy)
            betting_strategy.display_results()

        print("✅ 快速测试完成!")

    except Exception as e:
        print(f"❌ 快速测试失败: {e}")
        import traceback
        traceback.print_exc()
        raise


def demo_with_sample_data():
    """使用样本数据演示"""
    print("🎬 运行样本数据演示...")

    # 创建样本数据
    from data_inspector import create_sample_data_with_odds
    sample_data = create_sample_data_with_odds(200)

    # 保存样本数据
    sample_dir = 'data/sample'
    os.makedirs(sample_dir, exist_ok=True)
    sample_file = f'{sample_dir}/E0.csv'
    sample_data.to_csv(sample_file, index=False)
    print(f"✅ 样本数据已保存到: {sample_file}")

    # 运行演示
    demo_args = [
        '--country', 'England',
        '--start_season', '20',
        '--end_season', '20',
        '--model_name', 'LogisticRegression',
        '--evaluate_ckfm',
        '--do_value_betting',
        '--verbose'
    ]

    kwargs = parse_arguments(demo_args)
    kwargs['generate_mock_odds'] = False  # 使用真实样本赔率

    try:
        # 临时修改数据路径到样本目录
        original_path = 'data/2020'
        os.makedirs(original_path, exist_ok=True)

        import shutil
        shutil.copy(sample_file, f'{original_path}/E0.csv')

        league = League(betting_platforms, **kwargs)
        inspect_league_data(league)

        league.run()

        results_predictor = ResultsPredictor(league, **kwargs)
        results_predictor.train()
        results_predictor.eval()

        betting_strategy = BettingStrategy(results_predictor, **kwargs)
        league.seasons[-1].run(betting_strategy)
        betting_strategy.display_results()

        print("✅ 样本数据演示完成!")

    except Exception as e:
        print(f"❌ 样本数据演示失败: {e}")
        import traceback
        traceback.print_exc()


def show_help():
    """显示帮助信息"""
    print("""
🏈 体育博彩机器学习预测系统

使用方法:
  python main.py [选项]

快速开始:
  python main.py --quick_test              # 快速测试
  python main.py --demo                    # 样本数据演示

基本使用:
  python main.py --country England --start_season 18 --end_season 20

完整示例:
  python main.py \\
    --country England \\
    --start_season 18 \\
    --end_season 20 \\
    --model_name RandomForestClassifier \\
    --evaluate_ckfm \\
    --do_value_betting \\
    --generate_mock_odds

主要选项:
  --country              国家 (Italy, France, Spain, England, Germany)
  --start_season         开始赛季 (两位数年份, 如18表示2018-19赛季)
  --end_season           结束赛季 (可选, 默认为开始赛季的下一赛季)
  --model_name           模型类型 (LogisticRegression, RandomForestClassifier等)
  --evaluate_ckfm        启用CKFM价值投注评估
  --do_value_betting     启用价值投注策略
  --generate_mock_odds   如果缺少赔率数据则生成模拟数据
  --verbose              详细输出模式
  --inspect_data         检查数据质量

更多选项请运行: python main.py --help
""")


if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        show_help()
    elif '--quick_test' in sys.argv:
        quick_test()
    elif '--demo' in sys.argv:
        demo_with_sample_data()
    elif '--help' in sys.argv or '-h' in sys.argv:
        show_help()
        parse_arguments(['--help'])  # 显示完整帮助
    else:
        main()
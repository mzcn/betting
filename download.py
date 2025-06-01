#!/usr/bin/env python3
"""
下载真实足球数据的脚本
从 football-data.co.uk 下载历史比赛数据
"""

import os
import requests
import pandas as pd
from pathlib import Path
import time
from typing import List, Tuple


def download_season_data(country: str, division: int, season: str) -> bool:
    """
    下载指定联赛和赛季的数据

    Args:
        country: 国家名称 (England, Spain, Italy, Germany, France)
        division: 联赛级别 (1, 2, 3等)
        season: 赛季 (如 '1920' 表示 2019/2020赛季)

    Returns:
        bool: 是否成功下载
    """
    # 确定联赛代码
    league_codes = {
        'England': 'E',
        'Spain': 'SP',
        'Italy': 'I',
        'Germany': 'D',
        'France': 'F'
    }

    if country not in league_codes:
        print(f"❌ 不支持的国家: {country}")
        return False

    # 构建联赛标识符
    league_code = league_codes[country]
    if country == 'England':
        division -= 1  # 英格兰的编号比其他国家少1

    league_id = f"{league_code}{division}"

    # 构建URL
    base_url = "https://www.football-data.co.uk/mmz4281"
    file_url = f"{base_url}/{season}/{league_id}.csv"

    # 创建目录
    data_dir = Path(f"data/{season}")
    data_dir.mkdir(parents=True, exist_ok=True)

    # 下载文件
    local_file = data_dir / f"{league_id}.csv"

    if local_file.exists():
        print(f"✅ 文件已存在: {local_file}")
        return True

    print(f"📥 下载中: {file_url}")

    try:
        response = requests.get(file_url, timeout=30)
        response.raise_for_status()

        # 保存文件
        with open(local_file, 'wb') as f:
            f.write(response.content)

        # 验证数据
        df = pd.read_csv(local_file)
        if len(df) > 0:
            print(f"✅ 成功下载: {local_file} ({len(df)} 场比赛)")
            return True
        else:
            os.remove(local_file)
            print(f"❌ 数据为空: {local_file}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"❌ 下载失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        if local_file.exists():
            os.remove(local_file)
        return False


def download_multiple_seasons(country: str, division: int,
                              start_season: str, end_season: str) -> List[str]:
    """
    下载多个赛季的数据

    Args:
        country: 国家名称
        division: 联赛级别
        start_season: 开始赛季 (如 '18' 表示 2018/19)
        end_season: 结束赛季 (如 '20' 表示 2020/21)

    Returns:
        List[str]: 成功下载的赛季列表
    """
    start_year = int(start_season)
    end_year = int(end_season)

    if start_year > end_year:
        end_year += 100

    successful_seasons = []

    for year in range(start_year, end_year + 1):
        season_id = f'{year % 100:02d}{(year + 1) % 100:02d}'

        print(f"\n🗓️ 处理赛季 {season_id}...")

        if download_season_data(country, division, season_id):
            successful_seasons.append(season_id)
            time.sleep(1)  # 礼貌地等待，避免过快请求
        else:
            print(f"⚠️ 跳过赛季 {season_id}")

    return successful_seasons


def check_data_quality(country: str, division: int, season: str) -> Tuple[bool, dict]:
    """
    检查下载数据的质量

    Returns:
        Tuple[bool, dict]: (数据是否可用, 数据统计信息)
    """
    # 确定文件路径
    league_codes = {
        'England': 'E',
        'Spain': 'SP',
        'Italy': 'I',
        'Germany': 'D',
        'France': 'F'
    }

    league_code = league_codes[country]
    if country == 'England':
        division -= 1

    league_id = f"{league_code}{division}"
    file_path = f"data/{season}/{league_id}.csv"

    if not os.path.exists(file_path):
        return False, {"error": "文件不存在"}

    try:
        df = pd.read_csv(file_path)

        # 检查必要列
        required_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTR', 'FTHG', 'FTAG']
        missing_cols = [col for col in required_cols if col not in df.columns]

        # 检查赔率数据
        betting_platforms = ['B365', 'BW', 'IW', 'PS', 'WH', 'VC']
        odds_availability = {}

        for platform in betting_platforms:
            odds_cols = [f"{platform}{suffix}" for suffix in ['H', 'D', 'A']]
            if all(col in df.columns for col in odds_cols):
                non_null_count = df[odds_cols].notna().all(axis=1).sum()
                odds_availability[platform] = {
                    'available': True,
                    'coverage': non_null_count / len(df) * 100
                }
            else:
                odds_availability[platform] = {
                    'available': False,
                    'coverage': 0
                }

        stats = {
            'matches': len(df),
            'teams': df['HomeTeam'].nunique(),
            'missing_basic_cols': missing_cols,
            'odds_availability': odds_availability,
            'date_range': f"{df['Date'].iloc[0]} - {df['Date'].iloc[-1]}" if 'Date' in df.columns else "未知"
        }

        is_usable = len(missing_cols) == 0 and len(df) > 0

        return is_usable, stats

    except Exception as e:
        return False, {"error": str(e)}


def download_recommended_data():
    """
    下载推荐的数据集（用于测试和演示）
    """
    print("📊 开始下载推荐的数据集...")
    print("=" * 50)

    recommendations = [
        # 英超最近几个赛季
        ("England", 1, ["18", "19", "20", "21", "22"]),
        # 西甲
        ("Spain", 1, ["19", "20", "21"]),
        # 德甲
        ("Germany", 1, ["19", "20", "21"]),
    ]

    all_successful = []

    for country, division, seasons in recommendations:
        print(f"\n🏆 {country} - 第{division}级别联赛")
        print("-" * 30)

        for season in seasons:
            season_id = f'{season}{(int(season) + 1) % 100:02d}'
            if download_season_data(country, division, season_id):
                all_successful.append((country, division, season_id))

                # 检查数据质量
                is_usable, stats = check_data_quality(country, division, season_id)
                if is_usable:
                    print(f"   📊 数据统计: {stats['matches']} 场比赛, {stats['teams']} 支球队")

                    # 显示赔率覆盖率
                    best_platform = None
                    best_coverage = 0
                    for platform, info in stats['odds_availability'].items():
                        if info['available'] and info['coverage'] > best_coverage:
                            best_platform = platform
                            best_coverage = info['coverage']

                    if best_platform:
                        print(f"   💰 最佳赔率平台: {best_platform} (覆盖率 {best_coverage:.1f}%)")
                else:
                    print(f"   ⚠️ 数据质量问题: {stats}")

                time.sleep(1)

    # 总结
    print("\n" + "=" * 50)
    print("📊 下载总结")
    print("=" * 50)
    print(f"成功下载 {len(all_successful)} 个赛季的数据:")

    for country, division, season in all_successful:
        print(f"  ✅ {country} 第{division}级别 - {season} 赛季")

    return all_successful


def main():
    """
    主函数 - 交互式下载数据
    """
    print("🏈 足球数据下载工具")
    print("=" * 50)

    print("\n选择操作:")
    print("1. 下载推荐数据集（快速开始）")
    print("2. 自定义下载")
    print("3. 检查已有数据")

    choice = input("\n请选择 (1-3): ").strip()

    if choice == '1':
        download_recommended_data()

    elif choice == '2':
        # 自定义下载
        print("\n可用国家: England, Spain, Italy, Germany, France")
        country = input("国家: ").strip()

        division = int(input("联赛级别 (1-4): ").strip())

        start_season = input("开始赛季 (如18): ").strip()
        end_season = input("结束赛季 (如20): ").strip()

        successful = download_multiple_seasons(country, division, start_season, end_season)

        print(f"\n成功下载 {len(successful)} 个赛季")

    elif choice == '3':
        # 检查已有数据
        data_dir = Path("data")
        if not data_dir.exists():
            print("❌ 没有找到数据目录")
            return

        print("\n已有数据:")
        for season_dir in sorted(data_dir.iterdir()):
            if season_dir.is_dir():
                print(f"\n📁 赛季 {season_dir.name}:")
                for csv_file in season_dir.glob("*.csv"):
                    df = pd.read_csv(csv_file)
                    print(f"   - {csv_file.name}: {len(df)} 场比赛")

    else:
        print("❌ 无效选择")


if __name__ == "__main__":
    main()
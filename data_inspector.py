#!/usr/bin/env python3
"""
测试数据检查工具的修复版本
"""

import pandas as pd
import numpy as np
import os


def test_create_sample_data():
    """测试创建样本数据功能"""
    print("🧪 测试创建样本数据...")

    try:
        from data_inspector import create_sample_data_with_odds

        # 创建小量样本数据进行测试
        sample_data = create_sample_data_with_odds(20)

        print(f"✅ 成功创建 {len(sample_data)} 条样本数据")
        print(f"📊 数据列: {list(sample_data.columns)}")

        # 检查必要的列
        required_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTR', 'FTHG', 'FTAG', 'B365H', 'B365D', 'B365A']
        missing_cols = [col for col in required_cols if col not in sample_data.columns]

        if missing_cols:
            print(f"❌ 缺少必要列: {missing_cols}")
            return False
        else:
            print("✅ 所有必要列都存在")

        # 检查赔率数据
        odds_cols = ['B365H', 'B365D', 'B365A']
        for col in odds_cols:
            odds_values = sample_data[col]
            if odds_values.min() < 1.0 or odds_values.max() > 20.0:
                print(f"⚠️ {col} 赔率范围异常: {odds_values.min():.2f} - {odds_values.max():.2f}")
            else:
                print(f"✅ {col} 赔率范围正常: {odds_values.min():.2f} - {odds_values.max():.2f}")

        # 显示样本数据
        print("\n📋 样本数据预览:")
        print(sample_data.head(3).to_string())

        return True

    except Exception as e:
        print(f"❌ 创建样本数据失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_patch_odds():
    """测试添加模拟赔率功能"""
    print("\n🧪 测试添加模拟赔率...")

    try:
        from data_inspector import patch_dataset_with_mock_odds

        # 创建不完整的数据集
        incomplete_data = pd.DataFrame({
            'Date': ['01/01/2020', '02/01/2020', '03/01/2020'],
            'HomeTeam': ['Arsenal', 'Chelsea', 'Liverpool'],
            'AwayTeam': ['Chelsea', 'Liverpool', 'Arsenal'],
            'FTR': ['H', 'D', 'A'],
            'FTHG': [2, 1, 0],
            'FTAG': [1, 1, 2]
        })

        print(f"📊 原始数据列: {list(incomplete_data.columns)}")

        # 添加模拟赔率
        complete_data = patch_dataset_with_mock_odds(incomplete_data, 'B365')

        print(f"📊 处理后数据列: {list(complete_data.columns)}")

        # 验证赔率列
        odds_cols = ['B365H', 'B365D', 'B365A']
        for col in odds_cols:
            if col in complete_data.columns:
                non_null_count = complete_data[col].notna().sum()
                print(f"✅ {col}: {non_null_count}/{len(complete_data)} 非空")
            else:
                print(f"❌ 缺少 {col} 列")
                return False

        print("\n📋 处理后数据预览:")
        print(complete_data.to_string())

        return True

    except Exception as e:
        print(f"❌ 添加模拟赔率失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_inspection():
    """测试数据检查功能"""
    print("\n🧪 测试数据检查...")

    try:
        from data_inspector import inspect_dataset_columns, validate_odds_data

        # 创建测试数据
        test_data = pd.DataFrame({
            'Date': ['01/01/2020'] * 5,
            'HomeTeam': ['Team A'] * 5,
            'AwayTeam': ['Team B'] * 5,
            'FTR': ['H', 'D', 'A', 'H', 'D'],
            'B365H': [2.0, 3.0, 1.5, 2.5, 4.0],
            'B365D': [3.0, 3.2, 4.0, 3.1, 3.5],
            'B365A': [3.5, 2.1, 6.0, 2.8, 1.8]
        })

        # 检查数据
        info = inspect_dataset_columns(test_data, "测试数据")

        # 验证赔率
        odds_valid = validate_odds_data(test_data, ['B365H', 'B365D', 'B365A'])

        if odds_valid:
            print("✅ 赔率数据验证通过")
        else:
            print("❌ 赔率数据验证失败")
            return False

        return True

    except Exception as e:
        print(f"❌ 数据检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_save_sample_data():
    """测试保存样本数据"""
    print("\n🧪 测试保存样本数据...")

    try:
        from data_inspector import save_sample_data_to_file

        # 确保测试目录存在
        test_dir = 'test_data'
        os.makedirs(test_dir, exist_ok=True)

        # 保存样本数据
        filename = f'{test_dir}/test_sample.csv'
        saved_file = save_sample_data_to_file(filename, 10)

        # 验证文件是否创建
        if os.path.exists(saved_file):
            print(f"✅ 文件成功保存: {saved_file}")

            # 读取并验证
            loaded_data = pd.read_csv(saved_file)
            print(f"📊 加载数据: {len(loaded_data)} 行, {len(loaded_data.columns)} 列")

            # 清理测试文件
            os.remove(saved_file)
            os.rmdir(test_dir)
            print("🗑️ 清理测试文件")

            return True
        else:
            print(f"❌ 文件保存失败: {saved_file}")
            return False

    except Exception as e:
        print(f"❌ 保存样本数据失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("🏈 数据检查工具测试套件")
    print("=" * 50)

    tests = [
        ("创建样本数据", test_create_sample_data),
        ("添加模拟赔率", test_patch_odds),
        ("数据检查功能", test_data_inspection),
        ("保存样本数据", test_save_sample_data)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🧪 正在运行: {test_name}")
        print("-" * 30)

        try:
            if test_func():
                print(f"✅ {test_name} - 通过")
                passed += 1
            else:
                print(f"❌ {test_name} - 失败")
        except Exception as e:
            print(f"💥 {test_name} - 异常: {e}")

    print(f"\n📊 测试结果: {passed}/{total} 通过")

    if passed == total:
        print("🎉 所有测试通过! 数据检查工具工作正常")
        return True
    else:
        print("⚠️ 部分测试失败，请检查错误信息")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
#!/usr/bin/env python3
"""
便捷运行脚本 - 提供多种运行模式
"""

import sys
import os
import argparse
from datetime import datetime


def print_banner():
    """打印横幅"""
    print("""
🏈 =====================================================
   体育博彩机器学习预测系统
   Sports Betting ML Prediction System
   =====================================================
""")


def run_quick_test():
    """运行快速测试"""
    print("🧪 启动快速测试模式...")
    try:
        from main import quick_test
        quick_test()
    except ImportError:
        print("⚠️ 找不到 main 模块的 quick_test 函数，尝试使用替代方案...")
        run_quick_test_alternative()
    except Exception as e:
        print(f"❌ 快速测试失败: {e}")
        import traceback
        traceback.print_exc()


def run_quick_test_alternative():
    """快速测试的替代实现"""
    print("🧪 使用替代快速测试方案...")

    try:
        # 首先测试数据检查工具
        print("\n1️⃣ 测试数据检查工具...")
        from data_inspector import create_sample_data_with_odds, inspect_dataset_columns

        sample_data = create_sample_data_with_odds(20)
        print(f"✅ 成功创建 {len(sample_data)} 条测试数据")

        # 检查数据质量
        inspect_dataset_columns(sample_data, "快速测试数据")

        print("\n2️⃣ 测试核心模块导入...")
        try:
            from game import League
            from predictions import ResultsPredictor
            from betting import BettingStrategy
            print("✅ 核心模块导入成功")
        except Exception as e:
            print(f"❌ 核心模块导入失败: {e}")
            return

        print("\n3️⃣ 创建测试数据文件...")
        import os
        test_dir = 'data/1920'
        os.makedirs(test_dir, exist_ok=True)
        test_file = f'{test_dir}/E0.csv'
        sample_data.to_csv(test_file, index=False)
        print(f"✅ 测试数据已保存到: {test_file}")

        print("\n🎉 快速测试的基础准备工作完成！")
        print("💡 您现在可以尝试运行:")
        print("   python main.py --country England --start_season 19 --end_season 19 --evaluate_ckfm")

    except Exception as e:
        print(f"❌ 替代快速测试也失败了: {e}")
        import traceback
        traceback.print_exc()


def run_demo():
    """运行演示模式"""
    print("🎬 启动演示模式...")
    try:
        from main import demo_with_sample_data
        demo_with_sample_data()
    except (ImportError, AttributeError):
        print("⚠️ 找不到 main 模块的演示功能，使用替代方案...")
        run_demo_alternative()
    except Exception as e:
        print(f"❌ 演示模式失败: {e}")
        import traceback
        traceback.print_exc()


def run_demo_alternative():
    """演示模式的替代实现"""
    print("🎬 使用替代演示方案...")

    try:
        # 创建演示数据
        from data_inspector import create_sample_data_with_odds, patch_dataset_with_mock_odds
        import os

        print("📊 创建演示数据...")
        demo_data = create_sample_data_with_odds(50)

        # 保存演示数据
        demo_dir = 'data/demo'
        os.makedirs(demo_dir, exist_ok=True)
        demo_file = f'{demo_dir}/demo_matches.csv'
        demo_data.to_csv(demo_file, index=False)

        print(f"✅ 演示数据已保存到: {demo_file}")

        # 显示数据摘要
        from data_inspector import inspect_dataset_columns, validate_odds_data
        inspect_dataset_columns(demo_data, "演示数据")
        validate_odds_data(demo_data, ['B365H', 'B365D', 'B365A'])

        print("🎉 演示数据创建成功！您可以使用这些数据进行测试。")

    except Exception as e:
        print(f"❌ 替代演示方案也失败了: {e}")
        import traceback
        traceback.print_exc()


def run_data_check():
    """运行数据检查"""
    print("🔍 启动数据检查模式...")
    try:
        from data_inspector import save_sample_data_to_file, inspect_dataset_columns, validate_odds_data

        # 创建样本数据
        filename = save_sample_data_to_file('data/sample_check.csv', 100)

        # 检查数据
        import pandas as pd
        sample_data = pd.read_csv(filename)
        inspect_dataset_columns(sample_data, "样本数据检查")

        # 验证赔率数据
        print("\n🔍 验证赔率数据...")
        validate_odds_data(sample_data, ['B365H', 'B365D', 'B365A'])

        print(f"\n✅ 数据检查完成，样本文件保存在: {filename}")

    except Exception as e:
        print(f"❌ 数据检查失败: {e}")
        import traceback
        traceback.print_exc()


def run_full_analysis(country='England', start_season='18', end_season='20'):
    """运行完整分析"""
    print(f"📊 启动完整分析: {country} {start_season}-{end_season}")

    try:
        from main import main, parse_arguments

        args = [
            '--country', country,
            '--start_season', start_season,
            '--end_season', end_season,
            '--model_name', 'RandomForestClassifier',
            '--evaluate_ckfm',
            '--do_value_betting',
            '--verbose'
        ]

        # 临时替换sys.argv
        original_argv = sys.argv[:]
        sys.argv = ['main.py'] + args

        try:
            main()
        finally:
            sys.argv = original_argv

    except (ImportError, AttributeError):
        print("⚠️ 找不到 main 模块的完整功能，使用替代方案...")
        run_full_analysis_alternative(country, start_season, end_season)
    except Exception as e:
        print(f"❌ 完整分析失败: {e}")
        import traceback
        traceback.print_exc()


def run_full_analysis_alternative(country, start_season, end_season):
    """完整分析的替代实现"""
    print("📊 使用替代完整分析方案...")

    try:
        # 先准备测试数据
        print("📊 准备分析数据...")
        from data_inspector import create_sample_data_with_odds
        import os

        # 为每个赛季创建数据
        seasons = []
        start_year = int(start_season)
        end_year = int(end_season)

        if start_year > end_year:
            end_year += 100

        for year in range(start_year, end_year + 1):
            season_id = f'{year % 100:02d}{(year + 1) % 100:02d}'
            seasons.append(season_id)

        for season in seasons:
            # 创建季节数据目录
            season_dir = f'data/{season}'
            os.makedirs(season_dir, exist_ok=True)

            # 根据国家确定文件名
            if country.lower() == 'england':
                filename = 'E0.csv'
            elif country.lower() == 'spain':
                filename = 'SP1.csv'
            elif country.lower() == 'italy':
                filename = 'I1.csv'
            elif country.lower() == 'france':
                filename = 'F1.csv'
            elif country.lower() == 'germany':
                filename = 'D1.csv'
            else:
                filename = 'E0.csv'  # 默认英格兰

            season_file = f'{season_dir}/{filename}'

            if not os.path.exists(season_file):
                print(f"🔧 创建 {season} 赛季数据...")
                season_data = create_sample_data_with_odds(100)  # 每赛季100场比赛
                season_data.to_csv(season_file, index=False)
                print(f"✅ {season_file} 已创建")

        print(f"\n📊 数据准备完成，可以运行分析:")
        print(
            f"python main.py --country {country} --start_season {start_season} --end_season {end_season} --evaluate_ckfm --generate_mock_odds")

        # 尝试运行原始main.py
        print(f"\n🚀 尝试运行分析...")
        original_argv = sys.argv[:]
        sys.argv = [
            'main.py',
            '--country', country,
            '--start_season', start_season,
            '--end_season', end_season,
            '--model_name', 'LogisticRegression',  # 使用更简单的模型
            '--evaluate_ckfm'
        ]

        try:
            from main import main
            main()
        finally:
            sys.argv = original_argv

    except Exception as e:
        print(f"❌ 替代完整分析失败: {e}")
        import traceback
        traceback.print_exc()


def run_custom():
    """运行自定义配置"""
    print("⚙️ 自定义配置模式")

    countries = ['Italy', 'France', 'Spain', 'England', 'Germany']
    models = ['LogisticRegression', 'RandomForestClassifier', 'MLPClassifier', 'DecisionTreeClassifier']

    print("请选择配置:")

    # 选择国家
    print(f"\n国家选项: {', '.join(countries)}")
    country = input("选择国家 (默认England): ").strip() or 'England'
    if country not in countries:
        print(f"⚠️ 无效国家，使用默认值: England")
        country = 'England'

    # 选择赛季
    start_season = input("开始赛季 (如18表示2018-19, 默认18): ").strip() or '18'
    end_season = input("结束赛季 (默认20): ").strip() or '20'

    # 选择模型
    print(f"\n模型选项: {', '.join(models)}")
    model = input("选择模型 (默认LogisticRegression): ").strip() or 'LogisticRegression'
    if model not in models:
        print(f"⚠️ 无效模型，使用默认值: LogisticRegression")
        model = 'LogisticRegression'

    # 其他选项
    value_betting = input("启用价值投注? (y/N): ").strip().lower() in ['y', 'yes']
    ckfm_eval = input("启用CKFM评估? (Y/n): ").strip().lower() not in ['n', 'no']
    mock_odds = input("生成模拟赔率? (Y/n): ").strip().lower() not in ['n', 'no']

    print(f"\n📋 配置摘要:")
    print(f"   国家: {country}")
    print(f"   赛季: {start_season}-{end_season}")
    print(f"   模型: {model}")
    print(f"   价值投注: {value_betting}")
    print(f"   CKFM评估: {ckfm_eval}")
    print(f"   模拟赔率: {mock_odds}")

    confirm = input("\n确认运行? (Y/n): ").strip().lower() not in ['n', 'no']
    if not confirm:
        print("❌ 已取消")
        return

    try:
        # 构建参数
        args = [
            '--country', country,
            '--start_season', start_season,
            '--end_season', end_season,
            '--model_name', model,
            '--verbose'
        ]

        if value_betting:
            args.append('--do_value_betting')
        if ckfm_eval:
            args.append('--evaluate_ckfm')
        if mock_odds:
            args.append('--generate_mock_odds')

        # 首先尝试main
        try:
            from main import main
            print("🚀 使用主程序...")
        except ImportError:
            print("❌ 找不到可用的主程序模块")
            return

        # 临时替换sys.argv
        original_argv = sys.argv[:]
        sys.argv = ['main.py'] + args

        try:
            main()
        finally:
            sys.argv = original_argv

    except Exception as e:
        print(f"❌ 自定义配置运行失败: {e}")
        import traceback
        traceback.print_exc()


def main_menu():
    """主菜单"""
    print_banner()

    options = {
        '1': ('🧪 快速测试', run_quick_test),
        '2': ('🎬 演示模式', run_demo),
        '3': ('🔍 数据检查', run_data_check),
        '4': ('📊 完整分析 (England 18-20)', lambda: run_full_analysis()),
        '5': ('⚙️ 自定义配置', run_custom),
        '6': ('🛠️ 系统诊断', run_system_diagnosis),
        '7': ('❓ 帮助信息', show_help),
        '0': ('🚪 退出', None)
    }

    while True:
        print(f"\n请选择运行模式:")
        for key, (desc, _) in options.items():
            print(f"  {key}. {desc}")

        choice = input(f"\n请输入选项 (0-7): ").strip()

        if choice == '0':
            print("👋 再见!")
            break
        elif choice in options:
            desc, func = options[choice]
            if func:
                print(f"\n{desc}")
                print("-" * 50)
                try:
                    func()
                except KeyboardInterrupt:
                    print(f"\n❌ 操作被用户中断")
                except Exception as e:
                    print(f"\n💥 执行出错: {e}")
                    import traceback
                    traceback.print_exc()
                print(f"\n✅ {desc} 完成")
            else:
                break
        else:
            print(f"❌ 无效选项: {choice}")


def run_system_diagnosis():
    """运行系统诊断"""
    print("🛠️ 系统诊断模式")
    print("检查系统环境和依赖...")

    diagnosis_results = []

    # 1. 检查Python版本
    print("\n1️⃣ Python环境检查...")
    try:
        import sys
        python_version = sys.version
        print(f"✅ Python版本: {python_version.split()[0]}")
        diagnosis_results.append(("Python版本", "✅", python_version.split()[0]))
    except Exception as e:
        print(f"❌ Python环境检查失败: {e}")
        diagnosis_results.append(("Python版本", "❌", str(e)))

    # 2. 检查必要的库
    print("\n2️⃣ 依赖库检查...")
    required_libs = [
        'pandas', 'numpy', 'sklearn', 'matplotlib', 'seaborn'
    ]

    for lib in required_libs:
        try:
            __import__(lib)
            print(f"✅ {lib}: 已安装")
            diagnosis_results.append((f"库-{lib}", "✅", "已安装"))
        except ImportError:
            print(f"❌ {lib}: 未安装")
            diagnosis_results.append((f"库-{lib}", "❌", "未安装"))

    # 3. 检查项目文件
    print("\n3️⃣ 项目文件检查...")
    project_files = [
        'main.py', 'game.py', 'predictions.py', 'betting.py',
        'data_inspector.py', 'enhanced_main.py'
    ]

    import os
    for file in project_files:
        if os.path.exists(file):
            print(f"✅ {file}: 存在")
            diagnosis_results.append((f"文件-{file}", "✅", "存在"))
        else:
            print(f"❌ {file}: 缺失")
            diagnosis_results.append((f"文件-{file}", "❌", "缺失"))

    # 4. 检查数据目录
    print("\n4️⃣ 数据目录检查...")
    data_dirs = ['data', 'configs']
    for dir_name in data_dirs:
        if os.path.exists(dir_name):
            print(f"✅ {dir_name}/: 存在")
            diagnosis_results.append((f"目录-{dir_name}", "✅", "存在"))
        else:
            print(f"⚠️ {dir_name}/: 不存在 (将自动创建)")
            try:
                os.makedirs(dir_name, exist_ok=True)
                print(f"✅ {dir_name}/: 已创建")
                diagnosis_results.append((f"目录-{dir_name}", "✅", "已创建"))
            except Exception as e:
                print(f"❌ {dir_name}/: 创建失败 - {e}")
                diagnosis_results.append((f"目录-{dir_name}", "❌", f"创建失败: {e}"))

    # 5. 测试核心功能
    print("\n5️⃣ 核心功能测试...")
    try:
        from data_inspector import create_sample_data_with_odds
        test_data = create_sample_data_with_odds(5)
        print(f"✅ 数据生成功能: 正常 (生成{len(test_data)}条记录)")
        diagnosis_results.append(("数据生成功能", "✅", f"正常 (生成{len(test_data)}条记录)"))
    except Exception as e:
        print(f"❌ 数据生成功能: 失败 - {e}")
        diagnosis_results.append(("数据生成功能", "❌", f"失败: {e}"))

    # 6. 测试模块导入
    print("\n6️⃣ 模块导入测试...")
    core_modules = [
        ('game.League', 'game'),
        ('predictions.ResultsPredictor', 'predictions'),
        ('betting.BettingStrategy', 'betting')
    ]

    for module_class, module_name in core_modules:
        try:
            module = __import__(module_name)
            class_name = module_class.split('.')[1]
            getattr(module, class_name)
            print(f"✅ {module_class}: 可导入")
            diagnosis_results.append((f"模块-{module_class}", "✅", "可导入"))
        except Exception as e:
            print(f"❌ {module_class}: 导入失败 - {e}")
            diagnosis_results.append((f"模块-{module_class}", "❌", f"导入失败: {e}"))

    # 生成诊断报告
    print("\n" + "=" * 50)
    print("🔍 诊断报告摘要")
    print("=" * 50)

    success_count = sum(1 for _, status, _ in diagnosis_results if status == "✅")
    total_count = len(diagnosis_results)

    print(f"总检查项目: {total_count}")
    print(f"通过项目: {success_count}")
    print(f"失败项目: {total_count - success_count}")
    print(f"通过率: {success_count / total_count * 100:.1f}%")

    if success_count == total_count:
        print("\n🎉 系统状态良好！所有检查都通过了。")
        print("💡 您可以放心使用所有功能。")
    elif success_count >= total_count * 0.8:
        print("\n⚠️ 系统基本正常，但有少量问题。")
        print("💡 您可以尝试运行快速测试。")
    else:
        print("\n❌ 系统存在较多问题，建议先解决依赖和文件问题。")
        print("💡 请检查Python环境和必要的库安装。")

    # 提供建议
    print("\n📝 建议操作:")
    failed_libs = [item[0] for item in diagnosis_results
                   if item[0].startswith("库-") and item[1] == "❌"]
    if failed_libs:
        lib_names = [lib.replace("库-", "") for lib in failed_libs]
        print(f"   安装缺失的库: pip install {' '.join(lib_names)}")

    missing_files = [item[0] for item in diagnosis_results
                     if item[0].startswith("文件-") and item[1] == "❌"]
    if missing_files:
        print(f"   缺失的文件: {[f.replace('文件-', '') for f in missing_files]}")
        print("   请确保所有项目文件都在当前目录中。")

    return success_count >= total_count * 0.8


def show_help():
    """显示帮助信息"""
    print("""
📖 帮助信息
===========

运行模式说明:

1. 🧪 快速测试
   - 使用英格兰联赛单赛季数据
   - 自动生成模拟赔率
   - 快速验证系统功能

2. 🎬 演示模式  
   - 使用预生成的样本数据
   - 包含完整的赔率信息
   - 展示系统完整功能

3. 🔍 数据检查
   - 检查数据质量和完整性
   - 验证赔率数据有效性
   - 生成数据质量报告

4. 📊 完整分析
   - 使用多个赛季的真实数据
   - 完整的模型训练和评估
   - 投注策略回测

5. ⚙️ 自定义配置
   - 交互式配置选择
   - 自定义国家、赛季、模型
   - 灵活的参数设置

6. 🛠️ 系统诊断
   - 检查Python环境和依赖库
   - 验证项目文件完整性
   - 测试核心功能模块

命令行参数:
  python run.py --test        # 快速测试
  python run.py --demo        # 演示模式
  python run.py --check       # 数据检查
  python run.py --help        # 显示帮助

系统要求:
  - Python 3.7+
  - pandas, numpy, scikit-learn
  - matplotlib, seaborn (可视化)

故障排除:
  - 如果缺少赔率数据，使用 --generate_mock_odds 选项
  - 网络问题时，系统会使用本地缓存数据
  - 详细错误信息使用 --verbose 选项
  - 运行系统诊断检查环境: 选择选项 6

更多信息请查看项目文档。
""")
    """显示帮助信息"""
    print("""
📖 帮助信息
===========

运行模式说明:

1. 🧪 快速测试
   - 使用英格兰联赛单赛季数据
   - 自动生成模拟赔率
   - 快速验证系统功能

2. 🎬 演示模式  
   - 使用预生成的样本数据
   - 包含完整的赔率信息
   - 展示系统完整功能

3. 🔍 数据检查
   - 检查数据质量和完整性
   - 验证赔率数据有效性
   - 生成数据质量报告

4. 📊 完整分析
   - 使用多个赛季的真实数据
   - 完整的模型训练和评估
   - 投注策略回测

5. ⚙️ 自定义配置
   - 交互式配置选择
   - 自定义国家、赛季、模型
   - 灵活的参数设置

命令行参数:
  python run.py --test        # 快速测试
  python run.py --demo        # 演示模式
  python run.py --check       # 数据检查
  python run.py --help        # 显示帮助

系统要求:
  - Python 3.7+
  - pandas, numpy, scikit-learn
  - matplotlib, seaborn (可视化)

故障排除:
  - 如果缺少赔率数据，使用 --generate_mock_odds 选项
  - 网络问题时，系统会使用本地缓存数据
  - 详细错误信息使用 --verbose 选项

更多信息请查看项目文档。
""")


def parse_command_line():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="体育博彩ML系统便捷运行工具")

    parser.add_argument('--test', action='store_true', help='运行快速测试')
    parser.add_argument('--demo', action='store_true', help='运行演示模式')
    parser.add_argument('--check', action='store_true', help='运行数据检查')
    parser.add_argument('--full', action='store_true', help='运行完整分析')
    parser.add_argument('--country', default='England', help='国家选择')
    parser.add_argument('--start', default='18', help='开始赛季')
    parser.add_argument('--end', default='20', help='结束赛季')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_command_line()

    # 检查命令行参数
    if args.test:
        run_quick_test()
    elif args.demo:
        run_demo()
    elif args.check:
        run_data_check()
    elif args.full:
        run_full_analysis(args.country, args.start, args.end)
    else:
        # 没有命令行参数，启动交互式菜单
        main_menu()
# 文件路径: main.py
import pandas as pd
import warnings

# 1. 导入数据工具
from src.data.data_adapt import adapt_format
from src.data.data_check import check_df

# 2. 导入因子工厂
from src.factors.base import FACTOR_REGISTRY 
import src.factors.definitions  # 必须导入以触发注册

# 3. 导入处理器
from src.processor.cleaner import FactorCleaner
from src.processor.evaluate import FactorEvaluator

# 忽略 pandas 的一些未来版本警告
warnings.filterwarnings('ignore')

def main():
    print("量化因子挖掘启动...\n")

    # ==========================================
    # Step 1: 数据准备 (Data Preparation)
    # ==========================================
    print("[1/5] 读取与检查数据...")
    # 假设你的分钟数据路径
    data_path = '/Users/huoxubo/Quant/data/2025_stock_min_price.pq' # 请确保文件名正确
    try:
        df = pd.read_parquet(data_path, engine="fastparquet")
    except FileNotFoundError:
        print(f"❌ 错误：找不到文件 {data_path}")
        return

# 【新增】只取前 20000 行做测试！
    print("⚠️ 调试模式：仅使用前 1,000,000 行数据...")
    df = df.head(10000000).copy() 

    # 适配与检查
    print("   正在转换格式 (adapt_format)...")
    df = adapt_format(df)
    
    print("   正在检查排序 (check_df)...")
    df = check_df(df)
    print(df.head())
    # ...
    print(f"✅ 数据加载完成: {len(df)} 行, {df['asset'].nunique()} 只股票")

    # ==========================================
    # Step 2: 因子计算 (Factor Calculation)
    # ==========================================
    print("\n[2/5] 开始计算原始因子...")
    
    #在此配置你想挖掘的因子
    factor_config = [
        {"name": "RSI", "params": {"window": 10}, "shift": 1},  # 10分钟 RSI，是否滞后开关
        {"name": "ER", "params": {"window": 10}, "shift": 1},  # 10天路径效率，是否滞后开关
        {"name": "MACD", "params": {"fast": 12, "slow": 26, "signal": 9}, "shift": 1},  # MACD，是否滞后开关
        {"name": "PVT", "params": {}, "shift": 1},  # PVT，是否滞后开关
        {"name": "Turnover_Stability", "params": {"window": 10}, "shift": 1},  # 10天换手率稳定性，是否滞后开关
        {"name": "Ret_Turnover_Corr", "params": {"window": 10}, "shift": 1},  # 10天收益率与amount相关性，是否滞后开关
        {"name": "Capital_Gain_Overhang", "params": {"window": 10}, "shift": 1},  # 10天获利盘比例，是否滞后开关        
        {"name": "TSMOM", "params": {"window": 10}, "shift": 1},  # 10天时间序列动量
        {"name": "Volume_Price_Corr", "params": {"window": 10}, "shift": 1},  # 10天量价相关性，是否滞后开关
        {"name": "Individual_VOL", "params": {"window": 10}, "shift": 1},  # 10天剔除beta的波动率，是否滞后开关
        ]

    # 遍历配置，计算每个因子
    for config in factor_config:
        name = config['name']
        params = config['params']
        shift_steps = config.get('shift', 0)  # 默认不滞后

        if name not in FACTOR_REGISTRY:
            continue

        try:
            factor_cls = FACTOR_REGISTRY[name]
            instance = factor_cls(params)

            suffix = "_" + "_".join(str(v) for v in params.values()) if params else ""
            col_name = f"factor_{name}{suffix}"

            print(f"   -> 计算: {col_name}")
            raw_values = instance.calculate(df)
            df[col_name] = raw_values

        # 如果 shift_steps > 0，才做滞后
            if shift_steps > 0:
                df[col_name] = df.groupby('asset')[col_name].shift(shift_steps)

        except Exception as e:
            print(f"   ❌ {name} 计算失败: {e}")


    # ==========================================
    # Step 3: 因子清洗 (Factor Cleaning)
    # ==========================================
    print("\n[3/5] 开始因子清洗 (去极值/中性化/标准化)...")
    
    # 找到所有原始因子列
    raw_factors = [c for c in df.columns if c.startswith('factor_')]
    has_sector = 'sector' in df.columns # 检查是否有行业列
    
    for col in raw_factors:
        alpha_name = col.replace('factor_', 'alpha_')
        print(f"   -> 清洗: {col} => {alpha_name}")
        
        # 核心清洗步骤
        df[alpha_name] = FactorCleaner.process_factor(
            df, 
            col, 
            winsorize=False,    # 关闭去极值
            neutralize=False, # 如果有行业数据就做中性化，否则不做
            standardize=False, # 关闭标准化
            sector_col='sector'
        )

    # ==========================================
    # Step 4: 结果存档 (Persistence)
    # ==========================================
    print("\n[4/5] 保存 Alpha 因子库...")
    # 只保留 key columns 和 alpha columns
    final_cols = ['date', 'asset', 'close'] + [c for c in df.columns if c.startswith('alpha_')]
    df_alpha = df[final_cols].copy()
    
    save_path = "data/alpha_factors.csv"
    df_alpha.to_csv(save_path)
    print(f"✅ 文件已保存至: {save_path}")
    
# ==========================================
    # Step 5: 因子体检报告 & 结果存档
    # ==========================================
    print("\n[5/5] 生成因子体检报告 (Horizon=10min)...")
    
    # 1. 预处理
    df_eval = FactorEvaluator.preprocess_data(df_alpha, ret_col='next_ret', horizon=10)
    
    # 2. 找到所有 alpha 因子
    alpha_cols = [c for c in df_eval.columns if c.startswith('alpha_')]
    print(f"待评估因子: {alpha_cols}")

    summary_results = []

    # 3. 循环评估
    for factor in alpha_cols:
        print(f"\n{'='*60}")
        print(f"📊 因子: {factor}")
        print(f"{'='*60}")
        
        # --- A. IC 分析 ---
        ic_series = FactorEvaluator.calc_ic_series(df_eval, factor, 'next_ret')
        metrics = FactorEvaluator.calc_ic_metrics(ic_series)
        
        print(f"[1] IC 表现:")
        print(f"    IC均值: {metrics['IC_Mean']:.4f} | ICIR: {metrics['ICIR']:.4f} | 胜率: {metrics['Win_Rate']:.1%}")
        
        # --- B. Rolling IC ---
        rolling_ic = ic_series.rolling(window=20).mean()
        try:
            recent_trend = rolling_ic.tail(5).values.round(4) # 保留4位小数
            print(f"[2] 近期趋势 (Rolling IC): {recent_trend}")
        except:
            print("    (数据不足，无法计算 Rolling IC)")
        
        # --- C. Group Analysis ---
        group_rets = FactorEvaluator.calc_group_returns(df_eval, factor, 'next_ret')
        
        if group_rets.isnull().all():
            print("[3] 单调性: (数据不足)")
            continue

        ls_ret = group_rets.iloc[-1] - group_rets.iloc[0] # 多空收益
        
        print(f"[3] 分组收益 (单调性):")
        print(f"    多空收益 (Top-Bottom): {ls_ret*100:.3f}%")
        
        # 【修改点】不再画图，直接打印一个横向表格
        # 把 Series 转成 DataFrame 并转置(.T)，看起来像一行表格
        df_group_show = group_rets.to_frame(name='Avg_Ret').T
        print(df_group_show.round(6)) # 打印表格

        # --- D. 收集数据存 CSV ---
        record = {
            "Factor_Name": factor,
            "IC_Mean": metrics['IC_Mean'],
            "IC_Std": metrics['IC_Std'],
            "ICIR": metrics['ICIR'],
            "Win_Rate": metrics['Win_Rate'],
            "Long_Short_Ret": ls_ret,
        }
        for i, val in group_rets.items():
            record[f"Group_{i}_Ret"] = val
        summary_results.append(record)

    # --- 4. 保存结果 ---
    if summary_results:
        print("\n💾 正在保存评估汇总表...")
        df_report = pd.DataFrame(summary_results)
        df_report = df_report.sort_values(by="IC_Mean", ascending=False)
        
        save_path = "data/factor_report.csv"
        df_report.to_csv(save_path, index=False, float_format='%.6f')
        print(f"✅ 报告已保存: {save_path}")
    else:
        print("⚠️ 没有因子可以评估，报告未保存。")

    print("\n✅ 所有任务完成！")

if __name__ == "__main__":
    main()
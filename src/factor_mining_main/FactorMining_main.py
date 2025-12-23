# 文件路径: main.py
import pandas as pd
import warnings
import sys
import os

# 1. 获取当前脚本的绝对路径
current_path = os.path.dirname(os.path.abspath(__file__))
# 2. 向上回退两层，找到项目根目录 'Quant'
project_root = os.path.dirname(os.path.dirname(current_path))
# 3. 将根目录加入 python 搜索路径
if project_root not in sys.path:
    sys.path.append(project_root)

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
    df = df.head(100000).copy() 

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
# 在此配置你想挖掘的因子
    factor_config = [
        # ==========================
        # 1. 动量与趋势类 (Momentum & Trend)
        # ==========================
        {"name": "RSI", "params": {"window": 14}, "shift": 1},                 # 相对强弱指标
        {"name": "MACD", "params": {"fast": 12, "slow": 26, "signal": 9}, "shift": 1}, # MACD趋势
        {"name": "TSMOM", "params": {"window": 10}, "shift": 1},               # 时间序列动量
        {"name": "ROC", "params": {"window": 10}, "shift": 1},                 # 变动率 (Rate of Change)
        {"name": "BIAS", "params": {"window": 20}, "shift": 1},                # 乖离率 (价格偏离均线程度)
        {"name": "CCI", "params": {"window": 14}, "shift": 1},                 # 顺势指标 (需High/Low)
        {"name": "Aroon", "params": {"window": 25}, "shift": 1},               # 阿隆指标 (趋势强弱)
        {"name": "PriceRank", "params": {"window": 20}, "shift": 1},           # 价格在过去N天的分位数

        # ==========================
        # 2. 波动率与风险类 (Volatility & Risk)
        # ==========================
        {"name": "ATR", "params": {"window": 14}, "shift": 1},                 # 平均真实波幅 (绝对波动量)
        {"name": "Boll_Width", "params": {"window": 20}, "shift": 1},          # 布林带宽度 (波动率挤压)
        {"name": "Individual_VOL", "params": {"window": 20}, "shift": 1},      # 剔除Beta后的特异波动率
        {"name": "Return_Std", "params": {"window": 20}, "shift": 1},          # 简单收益率标准差
        {"name": "ER", "params": {"window": 10}, "shift": 1},                  # 路径效率 (卡夫曼效率系数)

        # ==========================
        # 3. 量价与资金流类 (Volume & Money Flow)
        # ==========================
        {"name": "PVT", "params": {}, "shift": 1},                             # 量价趋势指标
        {"name": "MFI", "params": {"window": 14}, "shift": 1},                 # 资金流量指标 (量化版RSI)
        {"name": "OBV", "params": {}, "shift": 1},                             # 能量潮 (需确认是否已注册，若无可用PVT代替)
        {"name": "VWAP_Bias", "params": {"window": 20}, "shift": 1},           # 价格对VWAP的偏离
        {"name": "VR", "params": {"window": 26}, "shift": 1},                  # 成交量比率
        {"name": "Volume_Price_Corr", "params": {"window": 10}, "shift": 1},   # 量价相关性

        # ==========================
        # 4. 情绪与反转类 (Sentiment & Reversal)
        # ==========================
        {"name": "WilliamsR", "params": {"window": 14}, "shift": 1},           # 威廉指标 (超买超卖)
        {"name": "PSY", "params": {"window": 12}, "shift": 1},                 # 心理线
        {"name": "Capital_Gain_Overhang", "params": {"window": 20}, "shift": 1}, # 获利盘比例 (CGO)

        # ==========================
        # 5. 流动性与微观结构 (Liquidity & Structure)
        # ==========================
        {"name": "Turnover_Stability", "params": {"window": 10}, "shift": 1},  # 换手率稳定性
        {"name": "Ret_Turnover_Corr", "params": {"window": 10}, "shift": 1},   # 收益率与换手率相关性
        {"name": "Amihud", "params": {"window": 20}, "shift": 1},              # 非流动性因子 (Amihud Illiquidity)
        
        # ==========================
        # 6. 统计分布特征 (Statistical)
        # ==========================
        {"name": "Skewness", "params": {"window": 20}, "shift": 1},            # 收益率分布偏度
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
        #rolling_ic = ic_series.rolling(window=20).mean()
        #try:
            #print(f"[2] 近期趋势 (Rolling IC): {recent_trend}")
        #except:
            #print("    (数据不足，无法计算 Rolling IC)")
        
        # --- C. Group Analysis ---
        avg_rets, cum_rets = FactorEvaluator.calc_group_returns(df_eval, factor, 'next_ret')
        
        if avg_rets.isnull().all():
            print("[3] 分组分析: (数据不足)")
            continue

        # 计算多空收益 (Group Top - Group Bottom)
        ls_avg = avg_rets.iloc[-1] - avg_rets.iloc[0]      # 平均多空

        print(f"[3] 分组收益 (Group Analysis):")
        print(f"    平均多空 (Avg Long-Short): {ls_avg*100:.4f}% (每期)")
        
        # 打印一个表格，包含两行：平均值 和 累计值
        # 组装成 DataFrame 方便打印
        df_show = pd.DataFrame({
            'Avg_Ret': avg_rets,       # 第一行：平均收益
            'Total_Cum': cum_rets      # 第二行：累计总收益
        }).T
        print(df_show.round(6)) 

        # --- D. 收集数据存 CSV ---
        record = {
            "Factor_Name": factor,
            "IC_Mean": metrics['IC_Mean'],
            "IC_Std": metrics['IC_Std'],
            "ICIR": metrics['ICIR'],
            "Win_Rate": metrics['Win_Rate'],
            # 保存多空数据
            "LS_Avg_Ret": ls_avg,
        }
        
        # 保存每一组的收益情况 (Avg 和 Cum 都存)
        for i in range(len(avg_rets)):
            record[f"G{i}_Avg"] = avg_rets.iloc[i]
        for i in range(len(avg_rets)):
            record[f"G{i}_Cum"] = cum_rets.iloc[i]
            
        summary_results.append(record)

    # 保存结果 (保持不变)
    if summary_results:
        print("\n💾 正在保存评估汇总表...")
        df_report = pd.DataFrame(summary_results)
        # 可以按 ICIR 或 累计多空收益 排序
        df_report = df_report.sort_values(by="IC_Mean", ascending=False)
        
        save_path = "data/factor_report.csv"
        df_report.to_csv(save_path, index=False, float_format='%.6f')
        print(f"✅ 报告已保存: {save_path}")
    else:
        print("⚠️ 没有因子可以评估，报告未保存。")

    print("\n✅ 所有任务完成！")

if __name__ == "__main__":
    main()
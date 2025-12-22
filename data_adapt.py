# src/data/processors/data_adapt.py
import pandas as pd
import numpy as np

def adapt_format(df):
    """
    【终极版】自动识别 + 排序 + 强力清洗 0 值 (防 inf)
    """
    print("   [Adapt] 开始数据适配...")
    df = df.copy()
    
    # 1. & 2. & 3. 这里的改名和时间合并逻辑保持不变...
    # (省略你原本写对的那些代码，直接保留即可)
    
    # ... (为了节省篇幅，假设中间代码和你发的一样) ...
    # 下面是关键修改部分：

    # ----------------------------------------------------
    # (把你原本的 Col Map 和 rename 代码放在这)
    col_map = {
        'Time': 'time', 'TIME': 'time', 'min_time': 'time',
        'Date': 'date', 'datetime': 'date',
        'code': 'asset', 'Ticker': 'asset',
        'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close',
        'Volume': 'volume', 'Vol': 'volume', 'vol': 'volume',
        'Turnover': 'turnover', 'Amount': 'amount', 'Amt': 'amount'
    }
    df = df.rename(columns=col_map)

    # ... (把你原本的时间合并代码放在这) ...
    if 'date' in df.columns and 'time' in df.columns:
        try:
            date_vals = pd.to_numeric(df['date'], errors='coerce').fillna(0).astype(np.int64)
            time_vals = pd.to_numeric(df['time'], errors='coerce').fillna(0).astype(np.int64)
            full_time_vals = date_vals * 10000 + time_vals
            df['date'] = pd.to_datetime(full_time_vals.astype(str), format='%Y%m%d%H%M')
        except Exception:
            pass

    # 4. 筛选列 (保留你的逻辑)
    wish_list = ['date', 'asset', 'open', 'high', 'low', 'close', 'volume', 'turnover', 'amount']
    final_cols = [c for c in wish_list if c in df.columns]
    df = df[final_cols]

    # 5. 排序 (必须先排序，才能做后面的填充！)
    print("   [Adapt] 正在排序...")
    df = df.sort_values(['asset', 'date']).reset_index(drop=True)

    # ========================================================
    # 🛑 新增步骤 6：强力清洗脏数据 (这才是解决 inf 的关键)
    # ========================================================
    print("   [Clean] 正在执行 0 值清洗和缺失填充...")

    # 定义哪些列绝对不能是 0
    price_cols = ['open', 'high', 'low', 'close']
    vol_cols = ['volume', 'turnover', 'amount']
    
    # 找到实际存在的列
    cols_to_clean = [c for c in price_cols + vol_cols if c in df.columns]

    # A. 将 0 和 inf 替换为 NaN
    # 这一步消灭了分母为0的可能性
    df[cols_to_clean] = df[cols_to_clean].replace([0, np.inf, -np.inf], np.nan)

    # B. 前向填充 (Forward Fill)
    # 逻辑：这分钟数据坏了，就沿用上一分钟的数据（假设价格没变）
    # 必须按 asset 分组填，防止股票A的数据填到股票B头上！
    df[cols_to_clean] = df.groupby('asset')[cols_to_clean].ffill()

    # C. 丢弃依然是 NaN 的行
    # 如果刚开盘就是 0 (前面没有数据可填)，这种数据彻底没救，删掉
    before_len = len(df)
    df = df.dropna(subset=['close']) # 只要 close 是空就删
    after_len = len(df)

    if before_len != after_len:
        print(f"   [Clean] 已剔除 {before_len - after_len} 行无法修复的脏数据")

    return df
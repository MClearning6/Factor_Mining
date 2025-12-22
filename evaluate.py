# 文件路径: src/processors/evaluator.py
import pandas as pd
import numpy as np

class FactorEvaluator:
    
    @staticmethod
    def preprocess_data(df: pd.DataFrame, ret_col='next_ret', horizon=5) -> pd.DataFrame:
        """
        准备数据：确保有未来收益率
        :param horizon: 预测未来第几根K线？
                        (日线通常用 1，分钟线建议用 5 或 10)
        """
        df = df.copy()
        # 如果还没算未来收益，先算一下
        if ret_col not in df.columns:
            # 【修改点】使用 horizon 而不是写死 -1
            # 逻辑：(未来第 N 根收盘价 / 当前收盘价) - 1
            df[ret_col] = df.groupby('asset')['close'].shift(-horizon) / df['close'] - 1

        df[ret_col] = df[ret_col].replace([np.inf, -np.inf], np.nan)

        # 必须去掉最后 horizon 行，否则 IC 是 NaN
        return df.dropna(subset=[ret_col])

    # ------------------------------------------------
    # 1. IC & Rolling IC (相关性 & 持续性)
    # ------------------------------------------------
    @staticmethod
    def calc_ic_series(df: pd.DataFrame, factor_col: str, ret_col: str) -> pd.Series:
        """
        计算每日 IC 序列
        """
        def daily_ic(group):
            # 分钟线切片可能数据很多，也可以保留 < 5 的判断
            if len(group) < 5: return np.nan 
            return group[factor_col].corr(group[ret_col], method='spearman')
        
        # include_groups=False 是新版 Pandas 的建议，不过为了兼容性先不加
        return df.groupby('date').apply(daily_ic)

    @staticmethod
    def calc_ic_metrics(ic_series: pd.Series) -> dict:
        """
        汇总 IC 指标
        """
        return {
            "IC_Mean": ic_series.mean(),
            "IC_Std":  ic_series.std(),
            # 年化系数：分钟线数据量大，这里暂时沿用 252 (日线逻辑) 或者根据分钟数调整
            # 如果是分钟线，ICIR 的绝对值意义不大，主要看相对大小
            "ICIR":    ic_series.mean() / ic_series.std() if ic_series.std() != 0 else 0,
            "Win_Rate": (ic_series > 0).mean() # 胜率
        }

    # ------------------------------------------------
    # 2. Group Return (单调性)
    # ------------------------------------------------
    @staticmethod
    def calc_group_returns(df: pd.DataFrame, factor_col: str, ret_col: str, n_bins=5) -> pd.Series:
        """
        分层回测：检查单调性
        """
        def get_group_ret(day_df):
            try:
                labels = list(range(n_bins))
                # duplicates='drop' 防止因子值大量重复导致分箱失败
                day_df['group'] = pd.qcut(day_df[factor_col], n_bins, labels=labels, duplicates='drop')
                return day_df.groupby('group')[ret_col].mean()
            except ValueError:
                return pd.Series(np.nan, index=range(n_bins))

        daily_group_rets = df.groupby('date').apply(get_group_ret)
        avg_group_rets = daily_group_rets.mean()
        
        return avg_group_rets
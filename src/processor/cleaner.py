import pandas as pd
import numpy as np

class FactorCleaner:
    
    @staticmethod
    def clean_inf(series: pd.Series) -> pd.Series:
        """仅将 inf 替换为 NaN，不填充"""
        return series.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def winsorize(series: pd.Series, limits=(0.01, 0.01)) -> pd.Series:
        """缩尾去极值：超出 1% 和 99% 的部分拉回边界"""
        # 必须先剔除 NaN 才能算分位，否则 quantile 可能会偏
        valid_series = series.dropna()
        if len(valid_series) == 0:
            return series
            
        q_min = valid_series.quantile(limits[0])
        q_max = valid_series.quantile(1.0 - limits[1])
        return series.clip(lower=q_min, upper=q_max)

    @staticmethod
    def z_score(series: pd.Series) -> pd.Series:
        """标准化：(x - mean) / std"""
        std = series.std()
        # 防止除以 0 产生新的 inf
        if std == 0 or np.isnan(std): 
            return pd.Series(0, index=series.index)
        return (series - series.mean()) / std

    @staticmethod
    def neutralize(df_group: pd.DataFrame, factor_col: str, sector_col: str) -> pd.Series:
        """行业中性化：减去行业均值"""
        if sector_col not in df_group.columns:
            return df_group[factor_col]
        
        # 使用 transform 计算行业均值，这比 apply 快很多
        sector_means = df_group.groupby(sector_col)[factor_col].transform('mean')
        
        # 结果 = 原始值 - 行业均值
        return df_group[factor_col] - sector_means

    # ==============================================
    # 入口函数
    # ==============================================
    @classmethod
    def process_factor(cls, df: pd.DataFrame, col_name: str, 
                       winsorize: bool = False,     # 默认关闭
                       neutralize: bool = False,   
                       standardize: bool = False,   # 默认关闭
                       sector_col: str = 'sector'
                       ) -> pd.Series:
        
        # 1. 预处理：先把 inf 变成 NaN
        s = cls.clean_inf(df[col_name])
        
        # 构造临时 DF 用于计算
        temp_df = df[['date']].copy()
        if neutralize and sector_col in df.columns:
            temp_df[sector_col] = df[sector_col]
        temp_df['val'] = s

        # 定义截面处理逻辑
        def cross_sectional_step(group):
            # A. 填充缺失值 (用当天的均值填充，比填 0 安全)
            # 如果整天都是 NaN，那就没办法了，填 0
            daily_mean = group['val'].mean()
            group['val'] = group['val'].fillna(daily_mean if not np.isnan(daily_mean) else 0)

            # B. 去极值 (Winsorize)
            if winsorize:
                group['val'] = cls.winsorize(group['val'])
            
            # C. 中性化 (Neutralize)
            if neutralize:
                group['val'] = cls.neutralize(group, 'val', sector_col)
            
            # D. 标准化 (Z-Score)
            if standardize:
                group['val'] = cls.z_score(group['val'])
            
            return group['val']

        # 2. 执行分组运算
        # group_keys=False 保证索引不被打乱
        processed_s = temp_df.groupby('date', group_keys=False).apply(cross_sectional_step)
        
        return processed_s
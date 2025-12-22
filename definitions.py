import pandas as pd
import numpy as np
# 从同级目录的 base.py 导入工具
from src.factors.base import FactorBase, register_factor

# ==========================================
# Part 1: 纯数学公式 (Math Logic)
# 这些函数只负责算数，不知道什么是因子，什么是股票
# ==========================================

def calc_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss.replace(0, np.nan) 
    return 100 - (100 / (1 + rs))

def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    diff = ema_fast - ema_slow
    dea = diff.ewm(span=signal, adjust=False).mean()
    macd = (diff - dea) * 2
    return macd

def calc_pvt(close, volume):
    ret = close.pct_change()
    pvt = (ret * volume).cumsum()
    return pvt

# 新增因子1：时间序列动量因子
def calc_ts_momentum(close, window=10):
    return close / close.shift(window)

# 新增因子2：路径效率
def calc_er(close, window=10):
    change = close.diff(window).abs()
    daily_volatility = close.diff().abs()
    path_length = daily_volatility.rolling(window).sum()
    er = change / path_length.replace(0, np.nan)
    return er.fillna(0)

# 新增因子3：剔除beta的波动率
def calc_individual_vol(close, window=10):
    ma = close.rolling(window).mean()
    bias = (close - ma) / ma
    ind_vol = bias.rolling(window).std()
    return -ind_vol

# 新增因子4：量价相关性（变化率 vs 变化率）
def calc_volume_price_corr(close, volume, window=10):
    delta_price = close.pct_change()
    delta_volume = volume.pct_change()
    corr = delta_price.rolling(window).corr(delta_volume)
    return corr.fillna(0)

# 新增因子5：收益率与活跃度匹配 (修正版)
# 【修复】增加 close 参数，修正变量引用错误
def calc_ret_turnover_corr(close, turnover, window=10):
    # 1. 计算收益率
    ret = close.pct_change()
    
    # 2. 计算相关系数 (收益率 vs 换手率绝对值)
    corr = ret.rolling(window).corr(turnover)
    
    return corr.fillna(0)

# 新增因子6：换手率稳定性 (修正版)
# 【修复】修正内部变量名为 turnover
def calc_turnover_stability(turnover, window=10):
    # 1. 计算均值和标准差
    mean_val = turnover.rolling(window).mean()
    std_val = turnover.rolling(window).std()
    
    # 2. 计算变异系数 (CV)
    cv = std_val / (mean_val + 1e-8)
    
    # 3. 取负号 (越稳越好)
    return -cv

# 新增因子7：获利盘比例 (修正版)
# 【修复】函数名改为 calc_cgo_math 以匹配类调用，参数改为接收计算好的滚动和
def calc_cgo_math(close, sum_volume, sum_turnover, window=10):
    # 1. 计算平均成本 (VWAP)
    avg_cost = sum_turnover / (sum_volume + 1e-8)
    
    # 2. 计算偏离度
    cgo = (close - avg_cost) / avg_cost
    
    return cgo.fillna(0)


# ==========================================
# Part 2: 因子类封装 (Factor Classes)
# ==========================================

# 新增因子7: 获利盘比例 CGO
@register_factor('Capital_Gain_Overhang')
class Capital_Gain_Overhang(FactorBase):
    @property
    def required_cols(self):
        return ['close', 'volume', 'turnover']

    def calculate(self, df):
        w = self.params.get('window', 10) # CGO通常周期较长，建议默认60
        
        # 分组计算滚动和
        sum_turnover = df.groupby('asset')['turnover'].transform(
            lambda x: x.rolling(window=w).sum()
        )
        sum_volume = df.groupby('asset')['volume'].transform(
            lambda x: x.rolling(window=w).sum()
        )
        # 调用纯数学逻辑
        return calc_cgo_math(df['close'], sum_volume, sum_turnover)

# 新增因子6: 换手率稳定性
@register_factor('Turnover_Stability')
class Turnover_Stability(FactorBase):
    @property
    def required_cols(self):
        return ['turnover']
    
    def calculate(self, df):
        w = self.params.get('window', 10)        
        return df.groupby('asset')['turnover'].transform(
            lambda x: calc_turnover_stability(x, window=w)
        )

# 新增因子5: 收益率与活跃度匹配
@register_factor('Ret_Turnover_Corr')
class Ret_Turnover_Corr(FactorBase):
    @property
    def required_cols(self):
        return ['close', 'turnover']

    def calculate(self, df):
        w = self.params.get('window', 10)
        
        def logic(sub_df):
            # 【修复】这里传入 sub_df['close']
            return calc_ret_turnover_corr(sub_df['close'], sub_df['turnover'], window=w)
            
        return df.groupby('asset', group_keys=False).apply(logic)

# 新增因子4: 量价相关性 (变化率版)
@register_factor('Volume_Price_Corr')
class Volume_Price_Corr(FactorBase):
    @property
    def required_cols(self):
        return ['close', 'volume']

    def calculate(self, df):
        w = self.params.get('window', 10)
        
        def logic(sub_df):
            return calc_volume_price_corr(sub_df['close'], sub_df['volume'], window=w)
            
        return df.groupby('asset', group_keys=False).apply(logic)

# 新增因子3：剔除beta的波动率
@register_factor('Individual_VOL')
class IndividualVolatility(FactorBase):
    @property
    def required_cols(self):
        return ['close']
    
    def calculate(self, df) -> pd.Series:
        w = self.params.get('window', 10)
        return df.groupby('asset')['close'].transform(
            lambda x: calc_individual_vol(x, window=w)
        )

# 新增因子2：路径效率
@register_factor('ER')
class EfficiencyRatio(FactorBase):
    @property
    def required_cols(self):
        return ['close']
    
    def calculate(self, df) -> pd.Series:
        w = self.params.get('window', 10)
        return df.groupby('asset')['close'].transform(
            lambda x: calc_er(x, window=w)
        )

# 新增因子1：时间序列动量
@register_factor("TSMOM")
class Momentum(FactorBase):
    @property
    def required_cols(self) -> list:
        return ['close']

    def calculate(self, df) -> pd.Series:
        w = self.params.get('window', 10)
        return df.groupby('asset')['close'].transform(
            lambda x: calc_ts_momentum(x, window=w)
        )

@register_factor('RSI')
class RSI(FactorBase):
    @property
    def required_cols(self):
        return ['close']
    
    def calculate(self, df) -> pd.Series:
        w = self.params.get('window', 10)
        return df.groupby('asset')['close'].transform(
            lambda x: calc_rsi(x, window=w)
        )

@register_factor("MACD")
class MACD(FactorBase):
    @property
    def required_cols(self) -> list:
        return ['close']

    def calculate(self, df) -> pd.Series:
        f = self.params.get('fast', 12)
        s = self.params.get('slow', 26)
        sig = self.params.get('signal', 9)
        return df.groupby('asset')['close'].transform(
            lambda x: calc_macd(x, fast=f, slow=s, signal=sig)
        )

@register_factor("PVT")
class PVT(FactorBase):
    @property
    def required_cols(self) -> list:
        return ['close', 'volume']

    def calculate(self, df) -> pd.Series:
        def apply_pvt(group):
            return calc_pvt(group['close'], group['volume'])
        result = df.groupby('asset', group_keys=False).apply(apply_pvt)
        return result
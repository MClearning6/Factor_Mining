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

#Gemini因子
# 11. BIAS 乖离率 (价格偏离均线的程度)
def calc_bias(close, window=20):
    ma = close.rolling(window).mean()
    return (close - ma) / (ma + 1e-8)

# 12. CCI 顺势指标 (需要 High, Low, Close)
def calc_cci(high, low, close, window=14):
    tp = (high + low + close) / 3
    ma = tp.rolling(window).mean()
    # Mean Absolute Deviation
    md = tp.rolling(window).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    cci = (tp - ma) / (0.015 * md.replace(0, np.nan))
    return cci

# 13. ATR 平均真实波幅 (衡量绝对波动量)
def calc_atr(high, low, close, window=14):
    c_prev = close.shift(1)
    tr1 = high - low
    tr2 = (high - c_prev).abs()
    tr3 = (low - c_prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window).mean()

# 14. Bollinger Band Width 布林带宽度 (衡量波动率的挤压与扩张)
def calc_boll_width(close, window=20, k=2):
    ma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = ma + k * std
    lower = ma - k * std
    # width = (upper - lower) / ma
    return (upper - lower) / ma

# 15. MFI 资金流量指标 (RSI的成交量加权版)
def calc_mfi(high, low, close, volume, window=14):
    tp = (high + low + close) / 3
    # 原始资金流 Raw Money Flow
    rmf = tp * volume
    
    delta = tp.diff()
    pos_flow = (rmf.where(delta > 0, 0)).rolling(window).sum()
    neg_flow = (rmf.where(delta < 0, 0)).rolling(window).sum()
    
    m_ratio = pos_flow / neg_flow.replace(0, np.nan)
    return 100 - (100 / (1 + m_ratio))

# 16. Williams %R 威廉指标 (衡量超买超卖)
def calc_willr(high, low, close, window=14):
    hh = high.rolling(window).max()
    ll = low.rolling(window).min()
    return -100 * (hh - close) / (hh - ll).replace(0, np.nan)

# 17. Amihud Illiquidity 非流动性因子 (收益率绝对值 / 成交额)
def calc_amihud(close, volume, window=20):
    ret_abs = close.pct_change().abs()
    amt = close * volume
    # 【修复】将0替换为NaN，或者加一个极小值避免除零
    amt = amt.replace(0, np.nan) 
    illip = (ret_abs / amt).rolling(window).mean()
    return illip * 1e6

# 18. Rolling Skewness 滚动偏度 (投资者通常偏好正偏度，厌恶负偏度)
def calc_skew(close, window=20):
    ret = close.pct_change()
    return ret.rolling(window).skew()

# 19. Rolling Rank 价格在过去N天的分位数 (0-1之间)
def calc_price_rank(close, window=20):
    return close.rolling(window).apply(lambda x: (x < x[-1]).sum() / (len(x) - 1), raw=True)

# 20. ROC 变动率 (动量的一种)
def calc_roc(close, window=12):
    return (close - close.shift(window)) / close.shift(window)

# 21. PSY 心理线 (上涨天数占比)
def calc_psy(close, window=12):
    diff = close.diff()
    up_days = (diff > 0).astype(int)
    return up_days.rolling(window).mean() * 100

# 22. VWAP Bias (收盘价相对于成交量加权均价的偏离)
def calc_vwap_bias(close, volume, window=20):
    # 近似计算滚动 VWAP
    cum_pv = (close * volume).rolling(window).sum()
    cum_v = volume.rolling(window).sum()
    vwap = cum_pv / cum_v.replace(0, np.nan)
    return (close / vwap) - 1

# 23. VR 容量比率 (上涨日成交量 / 下跌日成交量)
def calc_vr(close, volume, window=26):
    price_diff = close.diff()
    u_vol = volume.where(price_diff > 0, 0).rolling(window).sum()
    d_vol = volume.where(price_diff < 0, 0).rolling(window).sum()
    q_vol = volume.where(price_diff == 0, 0).rolling(window).sum()
    
    vr = (u_vol + 0.5 * q_vol) / (d_vol + 0.5 * q_vol + 1e-8)
    return vr * 100

# 24. STD 价格滚动标准差 (最朴素的波动率)
def calc_std(close, window=20):
    ret = close.pct_change()
    return ret.rolling(window).std()

# 25. Aroon Indicator (趋势强弱)
def calc_aroon(high, low, window=25):
    # 距离最高价的天数
    arg_max = high.rolling(window).apply(lambda x: x.argmax(), raw=True)
    # 距离最低价的天数
    arg_min = low.rolling(window).apply(lambda x: x.argmin(), raw=True)
    
    aroon_up = (arg_max + 1) / window * 100
    aroon_down = (arg_min + 1) / window * 100
    return aroon_up - aroon_down # 返回 Aroon Oscillator


# ==========================================
# Part 2: 因子类封装 (Factor Classes)
# ==========================================

#Gemini因子
@register_factor('BIAS')
class BIAS(FactorBase):
    @property
    def required_cols(self): return ['close']
    def calculate(self, df):
        w = self.params.get('window', 20)
        return df.groupby('asset')['close'].transform(lambda x: calc_bias(x, w))

@register_factor('CCI')
class CCI(FactorBase):
    @property
    def required_cols(self): return ['high', 'low', 'close']
    def calculate(self, df):
        w = self.params.get('window', 14)
        def logic(sub): return calc_cci(sub['high'], sub['low'], sub['close'], w)
        return df.groupby('asset', group_keys=False).apply(logic)

@register_factor('ATR')
class ATR(FactorBase):
    @property
    def required_cols(self): return ['high', 'low', 'close']
    def calculate(self, df):
        w = self.params.get('window', 14)
        def logic(sub): return calc_atr(sub['high'], sub['low'], sub['close'], w)
        return df.groupby('asset', group_keys=False).apply(logic)

@register_factor('Boll_Width')
class Boll_Width(FactorBase):
    @property
    def required_cols(self): return ['close']
    def calculate(self, df):
        w = self.params.get('window', 20)
        return df.groupby('asset')['close'].transform(lambda x: calc_boll_width(x, w))

@register_factor('MFI')
class MFI(FactorBase):
    @property
    def required_cols(self): return ['high', 'low', 'close', 'volume']
    def calculate(self, df):
        w = self.params.get('window', 14)
        def logic(sub): return calc_mfi(sub['high'], sub['low'], sub['close'], sub['volume'], w)
        return df.groupby('asset', group_keys=False).apply(logic)

@register_factor('WilliamsR')
class WilliamsR(FactorBase):
    @property
    def required_cols(self): return ['high', 'low', 'close']
    def calculate(self, df):
        w = self.params.get('window', 14)
        def logic(sub): return calc_willr(sub['high'], sub['low'], sub['close'], w)
        return df.groupby('asset', group_keys=False).apply(logic)

@register_factor('Amihud')
class Amihud(FactorBase):
    @property
    def required_cols(self): return ['close', 'volume']
    def calculate(self, df):
        w = self.params.get('window', 20)
        def logic(sub): return calc_amihud(sub['close'], sub['volume'], w)
        return df.groupby('asset', group_keys=False).apply(logic)

@register_factor('Skewness')
class Skewness(FactorBase):
    @property
    def required_cols(self): return ['close']
    def calculate(self, df):
        w = self.params.get('window', 20)
        return df.groupby('asset')['close'].transform(lambda x: calc_skew(x, w))

@register_factor('PriceRank')
class PriceRank(FactorBase):
    @property
    def required_cols(self): return ['close']
    def calculate(self, df):
        w = self.params.get('window', 20)
        return df.groupby('asset')['close'].transform(lambda x: calc_price_rank(x, w))

@register_factor('ROC')
class ROC(FactorBase):
    @property
    def required_cols(self): return ['close']
    def calculate(self, df):
        w = self.params.get('window', 12)
        return df.groupby('asset')['close'].transform(lambda x: calc_roc(x, w))

@register_factor('PSY')
class PSY(FactorBase):
    @property
    def required_cols(self): return ['close']
    def calculate(self, df):
        w = self.params.get('window', 12)
        return df.groupby('asset')['close'].transform(lambda x: calc_psy(x, w))

@register_factor('VWAP_Bias')
class VWAP_Bias(FactorBase):
    @property
    def required_cols(self): return ['close', 'volume']
    def calculate(self, df):
        w = self.params.get('window', 20)
        def logic(sub): return calc_vwap_bias(sub['close'], sub['volume'], w)
        return df.groupby('asset', group_keys=False).apply(logic)

@register_factor('VR')
class VR(FactorBase):
    @property
    def required_cols(self): return ['close', 'volume']
    def calculate(self, df):
        w = self.params.get('window', 26)
        def logic(sub): return calc_vr(sub['close'], sub['volume'], w)
        return df.groupby('asset', group_keys=False).apply(logic)

@register_factor('Return_Std')
class Return_Std(FactorBase):
    @property
    def required_cols(self): return ['close']
    def calculate(self, df):
        w = self.params.get('window', 20)
        return df.groupby('asset')['close'].transform(lambda x: calc_std(x, w))

@register_factor('Aroon')
class Aroon(FactorBase):
    @property
    def required_cols(self): return ['high', 'low']
    def calculate(self, df):
        w = self.params.get('window', 25)
        def logic(sub): return calc_aroon(sub['high'], sub['low'], w)
        return df.groupby('asset', group_keys=False).apply(logic)

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
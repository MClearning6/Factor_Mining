# 文件路径: src/factors/base.py
from abc import ABC, abstractmethod
import pandas as pd 

# --- 定义开始 ---

# 1. 这是一个全局字典（花名册）
# 用来存： "名字" -> "类" 的对应关系
FACTOR_REGISTRY = {} 

# 2. 这是装饰器的定义（注册员）
def register_factor(name: str):
    """
    这是一个闭包函数。
    name: 是你在 @register_factor("这里写的名字")
    """
    def decorator(cls):
        # cls: 就是被装饰的那个类 (比如 class RSI)
        
        # 核心动作：把名字和类存进字典
        FACTOR_REGISTRY[name] = cls 
        
        # 把类原封不动地返回去（不改变类的功能，只做记录）
        return cls
        
    return decorator

# --- 定义结束 ---

class FactorBase(ABC):
    def __init__(self, params: dict = None):
        self.params = params if params else {}
        self.name = self.__class__.__name__

    # 规定每个因子必须检查需要的变量
    @property
    @abstractmethod
    def required_cols(self) -> list:
        pass

    def check_df(self, df):
        # 这里的 columns 检查很关键
        missing = [col for col in self.required_cols if col not in df.columns]
        if missing:
            print(f'[{self.name}] 无法计算，缺少列：{missing}')
            return False

        return True

    @abstractmethod
    def calculate(self, df) -> pd.Series:
        pass
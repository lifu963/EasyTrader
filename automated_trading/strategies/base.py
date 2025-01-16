import pandas as pd
from abc import ABC, abstractmethod


class BaseInvestmentStrategy(ABC):
    """
    投资策略基类，所有策略都应继承该类并实现 apply 方法。
    """

    def __init__(self):
        self.name = ""  # 策略名称
        self.description = ""  # 策略原理
        self.parameters = {}  # 参数说明 {参数名: (说明, 类型, 默认值)}

    @abstractmethod
    def apply(self, data: pd.Series, **kwargs) -> pd.Series:
        """
        具体策略的实现逻辑
        :param data: 时间序列数据（资产价格）
        :return: 每日资产份额的时间序列
        """
        pass

    def get_strategy_info(self) -> str:
        info = [f"### *{self.name}*", "", f"{self.description}", "", "#### 参数说明："]
        for param_name, (desc, param_type, default) in self.parameters.items():
            info.append(f"- **{param_name}**：*{desc}*，{param_type.__name__} 类型，默认值 {default}")
        return "\n".join(info)

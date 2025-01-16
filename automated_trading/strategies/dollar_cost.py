import pandas as pd
from .base import BaseInvestmentStrategy
from .registry import StrategyRegistry


@StrategyRegistry.register
class DollarCostAveragingStrategy(BaseInvestmentStrategy):
    """
    定投策略实现。
    """

    def __init__(self):
        super().__init__()
        self.name = "定投策略"
        self.description = "每月固定日期投入固定金额购买资产，长期持有，以降低择时风险。"
        self.parameters = {
            'monthly_investment': ('每月定投金额', float, 2000)
        }

    def apply(self, data: pd.Series, initial_investment: float, monthly_investment: float = 2000) -> pd.Series:
        # 初始投资份额
        shares = initial_investment / data.iloc[0]
        daily_target_shares = [shares]

        for date, price in data.iloc[1:].items():
            if date.day == 1:  # 每月1日定投
                shares += monthly_investment / price
            daily_target_shares.append(shares)

        return pd.Series(daily_target_shares, index=data.index, name=data.name)

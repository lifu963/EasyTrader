import pandas as pd
from typing import Dict

ANNUAL_INFLATION_RATE = 0.07  # 年化通货膨胀率


class Metrics:
    """
    财务和投资指标的计算。
    """

    @staticmethod
    def calculate_inflation_benchmark(base_value: float,
                                      dates: pd.Index,
                                      annual_rate: float = ANNUAL_INFLATION_RATE) -> pd.Series:
        """
        计算通货膨胀基准线。
        """
        daily_rate = (1 + annual_rate) ** (1 / 365)
        days_elapsed = (dates - dates[0]).days
        values = base_value * (daily_rate ** days_elapsed)
        return pd.Series(values, index=dates)

    @staticmethod
    def calculate_max_drawdown(net_worth: pd.Series) -> Dict[str, float]:
        """
        计算最大回撤及相关信息。
        """
        if len(net_worth) < 2 or net_worth.isnull().all():
            return {
                'max_drawdown': 0.0,
                'start_date': None,
                'end_date': None,
            }

        rolling_max = net_worth.cummax()
        drawdown = (net_worth - rolling_max) / rolling_max

        if drawdown.isnull().all():
            return {
                'max_drawdown': 0.0,
                'start_date': None,
                'end_date': None,
            }

        max_drawdown = drawdown.min()
        end_date = drawdown.idxmin()
        start_date = rolling_max[:end_date].idxmax()

        return {
            'max_drawdown': max_drawdown,
            'start_date': start_date,
            'end_date': end_date,
        }

import pandas as pd
from .base import BaseInvestmentStrategy
from .registry import StrategyRegistry


@StrategyRegistry.register
class TrendFollowingStrategy(BaseInvestmentStrategy):
    """
    趋势跟随策略实现：通过长期均线判断趋势，在上升趋势确认时买入，下降趋势确认时卖出，以避免追涨杀跌。
    """

    def __init__(self):
        super().__init__()
        self.name = "趋势跟随策略"
        self.description = "通过长期均线判断趋势，在上升趋势确认时买入，下降趋势确认时卖出，以避免追涨杀跌。"
        self.parameters = {
            'long_window': ('长期均线窗口大小（天数）', int, 200),
            'confirmation_days': ('趋势确认的天数（连续上涨/下跌的天数）', int, 10)
        }

    def apply(self, data: pd.Series, initial_investment: float, long_window: int = 200, confirmation_days: int = 10) -> pd.Series:
        """
        策略实现逻辑。
        """
        # 计算长期均线
        long_ma = data.rolling(window=long_window).mean()

        # 初始化变量
        shares = 0  # 当前持仓（份额）
        cash = initial_investment  # 初始现金
        daily_target_shares = []  # 每日持仓记录
        last_signal = None  # 上一次的交易信号（"buy" 或 "sell"）

        # 遍历价格数据，逐日计算
        for i in range(len(data)):
            current_price = data.iloc[i]
            current_ma = long_ma.iloc[i] if i >= long_window else None

            # 趋势信号判断（确保足够数据可用）
            if i >= confirmation_days and current_ma is not None:
                recent_prices = data.iloc[i - confirmation_days + 1:i + 1]
                if current_price > current_ma and all(recent_prices > current_ma):
                    signal = "buy"
                elif current_price < current_ma and all(recent_prices < current_ma):
                    signal = "sell"
                else:
                    signal = None
            else:
                signal = None

            # 执行交易逻辑
            if signal == "buy" and last_signal != "buy":
                shares = cash / current_price  # 全仓买入
                cash = 0
                last_signal = "buy"
            elif signal == "sell" and last_signal != "sell":
                cash = shares * current_price  # 全仓卖出
                shares = 0
                last_signal = "sell"

            # 记录当天的持仓
            daily_target_shares.append(shares)

        # 返回每日持仓份额的时间序列
        return pd.Series(daily_target_shares, index=data.index)

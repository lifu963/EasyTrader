import pandas as pd
from typing import List, Dict, Tuple
import inspect

from automated_trading.strategies.base import BaseInvestmentStrategy


class BacktestSystem:
    """
    通用回测系统，支持动态策略类。
    """

    @staticmethod
    def filter_strategy_params(strategy: BaseInvestmentStrategy, params: dict) -> dict:
        """
        过滤出策略 apply 方法所需的参数。
        """
        # 获取 apply 方法的参数列表
        sig = inspect.signature(strategy.apply)
        # 获取所有参数名（不包括 self 和 data）
        valid_params = [param.name for param in sig.parameters.values()
                        if param.name not in ['self', 'data']]

        # 只保留策略所需的参数
        filtered_params = {k: v for k, v in params.items() if k in valid_params}
        return filtered_params

    @staticmethod
    def backtest(
            data: pd.Series,
            strategy: BaseInvestmentStrategy,
            start_date=None,
            end_date=None,
            **kwargs
    ) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        回测系统，计算投资结果。
        """
        # 过滤策略参数
        strategy_params = BacktestSystem.filter_strategy_params(strategy, kwargs)

        # 如果指定了时间范围，裁剪数据
        if start_date:
            start_date = pd.to_datetime(start_date)
            data = data[data.index >= start_date]
        if end_date:
            end_date = pd.to_datetime(end_date)
            data = data[data.index <= end_date]

        # 打印回测时间范围
        print(f"Backtesting from {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")

        # 初始化资金和持仓
        available_cash = 0  # 账户剩余金额
        holding_shares = 0  # 当前持仓（份额）
        cumulative_external_investment = 0  # 累计外部投入的资金总额

        # 记录每日状态
        net_worth_history = []  # 总资产（现金 + 持仓价值）
        external_investment_history = []  # 累计外部投资
        profit_history = []  # 盈亏记录
        dates = []
        trade_log = []  # 记录交易信息

        # 获取策略的每日目标持仓份额
        try:
            daily_target_shares = strategy.apply(data, **strategy_params)
        except (TypeError, ValueError) as e:
            raise ValueError(f"### 策略参数错误\n{str(e)}")

        # 回测逻辑
        for date, price in data.items():

            # 获取策略当天的目标持仓（份额）
            target_shares = daily_target_shares.loc[date]

            # 买入操作
            if target_shares > holding_shares:
                buy_shares = target_shares - holding_shares
                buy_cost = buy_shares * price
                # 如果现金不足，则从外部注入资金
                if buy_cost > available_cash:
                    external_investment = buy_cost - available_cash  # 需要从外部投入的资金
                    available_cash += external_investment  # 增加外部资金到现金
                    cumulative_external_investment += external_investment  # 累加外部投入资金
                # 扣除现金，增加持仓
                available_cash -= buy_cost
                holding_shares += buy_shares

                # 记录交易日志
                trade_log.append({
                    'date': date,
                    'action': 'buy',
                    'shares': buy_shares,
                    'price': price,
                    'amount': buy_cost
                })

            # 如果目标持仓低于当前持仓，执行卖出操作
            elif target_shares < holding_shares:
                sell_shares = holding_shares - target_shares
                sell_revenue = sell_shares * price
                available_cash += sell_revenue  # 增加卖出收入
                holding_shares -= sell_shares  # 减少持仓

                # 记录交易日志
                trade_log.append({
                    'date': date,
                    'action': 'sell',
                    'shares': sell_shares,
                    'price': price,
                    'amount': sell_revenue
                })

            # 更新每日状态
            holding_value = holding_shares * price
            net_worth = available_cash + holding_value
            profit = net_worth - cumulative_external_investment

            # 记录每日数据
            net_worth_history.append(net_worth)
            external_investment_history.append(cumulative_external_investment)
            profit_history.append(profit)
            dates.append(date)

        # 构建回测结果 DataFrame
        backtest_results = pd.DataFrame({
            'Total Investment': external_investment_history,
            'Net Worth': net_worth_history,
            'Profit': profit_history,
        }, index=dates)

        return backtest_results, trade_log

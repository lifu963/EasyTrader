import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple


class Visualization:
    """
    可视化工具。
    """

    @staticmethod
    def _setup_base_plot(title: str, xlabel: str = "Date", ylabel: str = "Value") -> Tuple[plt.Figure, plt.Axes]:
        """
        设置基础图表样式。
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title(title, fontsize=16, fontweight='bold', color='darkblue')
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.grid(which='major', linestyle='--', linewidth=0.5, alpha=0.7)
        return fig, ax

    @staticmethod
    def plot_price_with_benchmark(prices: pd.Series,
                                  benchmark: pd.Series,
                                  title: str,
                                  price_label: str = "Price",
                                  benchmark_label: str = "Benchmark") -> plt.Figure:
        """
        绘制价格和基准线对比图。
        """
        fig, ax = Visualization._setup_base_plot(title)

        ax.plot(prices.index, prices,
                label=price_label, color='black', alpha=0.6, linewidth=1)
        ax.plot(benchmark.index, benchmark,
                label=benchmark_label, color='red', linestyle=':')

        ax.legend()
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_backtest_results(results: pd.DataFrame, max_drawdown_info: Dict, title: str) -> plt.Figure:
        """
        绘制回测结果。
        """
        fig, ax = Visualization._setup_base_plot(title)

        # 绘制投资和净值曲线
        ax.plot(results.index, results['Total Investment'],
                label="Total Investment", color='blue', linestyle='--')
        ax.plot(results.index, results['Net Worth'],
                label="Net Worth", color='green', alpha=0.7)

        # 填充盈亏区域
        ax.fill_between(results.index, results['Total Investment'], results['Net Worth'],
                        where=results['Net Worth'] >= results['Total Investment'],
                        color='green', alpha=0.3, label="Profit")
        ax.fill_between(results.index, results['Total Investment'], results['Net Worth'],
                        where=results['Net Worth'] < results['Total Investment'],
                        color='red', alpha=0.3, label="Loss")

        # 标记最大回撤
        start_date = max_drawdown_info['start_date']
        end_date = max_drawdown_info['end_date']
        if start_date and end_date:
            ax.scatter([start_date], [results['Net Worth'].loc[start_date]],
                       color='orange', label="Max Drawdown Start", zorder=5, s=100, alpha=0.8)
            ax.scatter([end_date], [results['Net Worth'].loc[end_date]],
                       color='red', label="Max Drawdown End", zorder=5, s=100, alpha=0.8)
            ax.plot([start_date, end_date],
                    [results['Net Worth'].loc[start_date], results['Net Worth'].loc[end_date]],
                    color='red', linestyle='--',
                    label=f"Max Drawdown: {max_drawdown_info['max_drawdown']:.2%}")

        ax.legend(fontsize=10)
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_trades(prices: pd.Series,
                    trade_log: List[Dict],
                    title: str,
                    start_date: str = None,
                    end_date: str = None) -> plt.Figure:
        """
        绘制交易记录。
        """

        # 过滤价格数据
        if start_date:
            start_date = pd.to_datetime(start_date)
            prices = prices[prices.index >= start_date]
        if end_date:
            end_date = pd.to_datetime(end_date)
            prices = prices[prices.index <= end_date]

        fig, ax = Visualization._setup_base_plot(title)

        # 绘制价格曲线
        ax.plot(prices.index, prices, color='black', label="Price", alpha=0.7)

        # 过滤并绘制交易点
        filtered_trades = trade_log
        if start_date or end_date:
            filtered_trades = [
                trade for trade in trade_log
                if (not start_date or pd.to_datetime(trade['date']) >= start_date) and
                   (not end_date or pd.to_datetime(trade['date']) <= end_date)
            ]

        buy_trades = [(log['date'], log['price']) for log in filtered_trades if log['action'] == 'buy']
        sell_trades = [(log['date'], log['price']) for log in filtered_trades if log['action'] == 'sell']

        if buy_trades:
            buy_dates, buy_prices = zip(*buy_trades)
            ax.scatter(buy_dates, buy_prices, color='green', label='Buy', zorder=5, s=50, alpha=0.8)

        if sell_trades:
            sell_dates, sell_prices = zip(*sell_trades)
            ax.scatter(sell_dates, sell_prices, color='red', label='Sell', zorder=5, s=50, alpha=0.8)

        ax.legend(fontsize=12)
        fig.tight_layout()
        return fig

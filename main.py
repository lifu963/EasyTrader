import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Tuple

import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# === 常量定义 ===
CACHE_FILE = 'finance_data.csv'
TIMESTAMP_FILE = 'last_update.txt'
ANNUAL_INFLATION_RATE = 0.07  # 通货膨胀率
TIME_RANGES = {
    "全部": None,
    "5年": pd.DateOffset(years=5),
    "1年": pd.DateOffset(years=1),
    "6个月": pd.DateOffset(months=6),
    "1个月": pd.DateOffset(months=1)
}


class LocalFileHandler:
    """
    本地文件操作
    """

    @staticmethod
    def exists(filepath: str) -> bool:
        return os.path.exists(filepath)

    @staticmethod
    def read(filepath: str) -> str:
        with open(filepath, 'r') as f:
            return f.read()

    @staticmethod
    def write(filepath: str, content: str):
        with open(filepath, 'w') as f:
            f.write(content)


# === 数据加载模块 ===
class DataLoader:
    """
    数据加载与缓存管理，支持文件系统的依赖注入。
    """

    def __init__(self, symbols: List[str], cache_file: str = CACHE_FILE, timestamp_file: str = TIMESTAMP_FILE,
                 file_handler=None):
        self.symbols = symbols
        self.cache_file = cache_file
        self.timestamp_file = timestamp_file
        self.file_handler = file_handler or LocalFileHandler()  # 默认使用本地文件系统

    def is_data_outdated(self) -> bool:
        """
        检查缓存数据是否过期（一天更新一次）
        """
        if not self.file_handler.exists(self.timestamp_file):
            return True

        last_update = self.file_handler.read(self.timestamp_file).strip()
        today = datetime.now().strftime('%Y-%m-%d')
        return today != last_update

    def update_timestamp(self):
        """
        更新时间戳文件
        """
        today = datetime.now().strftime('%Y-%m-%d')
        self.file_handler.write(self.timestamp_file, today)

    def fetch_data(self) -> pd.DataFrame:
        """
        从缓存或网络加载数据
        """
        if self.file_handler.exists(self.cache_file) and not self.is_data_outdated():
            print("Loading data from cache...")
            return pd.read_csv(self.cache_file, index_col=0, parse_dates=True, header=[0, 1])

        print("Downloading new data...")

        start_date = datetime.strptime("2015-01-01", "%Y-%m-%d")
        end_date = datetime.now().strftime('%Y-%m-%d')

        symbols_str = " ".join(self.symbols)
        data = yf.download(symbols_str, start=start_date.strftime('%Y-%m-%d'), end=end_date, group_by='tickers')
        data.to_csv(self.cache_file)
        self.update_timestamp()

        return data


# === 指标计算模块 ===
class Metrics:
    """
    财务和投资指标的计算
    """

    @staticmethod
    def calculate_inflation_benchmark(base_value: float,
                                      dates: pd.Index,
                                      annual_rate: float = ANNUAL_INFLATION_RATE) -> pd.Series:
        """
        计算通货膨胀基准线
        :param base_value: 基准起始值
        :param dates: 日期索引
        :param annual_rate: 年化通胀率
        :return: 通胀基准线数据序列
        """
        daily_rate = (1 + annual_rate) ** (1 / 365)
        days_elapsed = (dates - dates[0]).days
        values = base_value * (daily_rate ** days_elapsed)
        return pd.Series(values, index=dates)

    @staticmethod
    def calculate_max_drawdown(net_worth: pd.Series) -> Dict[str, float]:
        """
        计算最大回撤及相关信息
        :param net_worth: 净值数据序列
        :return: 包含最大回撤值及其起点和终点日期的字典
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


class BaseInvestmentStrategy(ABC):
    """
    投资策略基类，所有策略都应继承该类并实现 apply 方法。
    """

    @abstractmethod
    def apply(self, data: pd.Series, **kwargs) -> pd.Series:
        """
        具体策略的实现逻辑
        :param data: 时间序列数据（资产价格）
        :return: 每日资产份额的时间序列
        """
        pass


class DollarCostAveragingStrategy(BaseInvestmentStrategy):
    """
    定投策略实现
    """
    def __init__(self, monthly_investment: int = 2000):
        self.monthly_investment = monthly_investment

    def apply(self, data: pd.Series, initial_investment: float) -> pd.Series:
        # 初始投资份额
        shares = initial_investment / data.iloc[0]
        daily_target_shares = [shares]

        for date in data.index[1:]:
            if date.day == 1:  # 每月1日定投
                shares += self.monthly_investment / data.loc[date]
            daily_target_shares.append(shares)

        return pd.Series(daily_target_shares, index=data.index)


class TrendFollowingStrategy(BaseInvestmentStrategy):
    """
    趋势跟随策略实现：不赚最后一个铜板
    """

    def __init__(self, long_window: int = 200, confirmation_days: int = 10):
        """
        :param long_window: 长期均线窗口大小（天数）
        :param confirmation_days: 趋势确认的天数（连续上涨/下跌的天数）
        """
        self.long_window = long_window
        self.confirmation_days = confirmation_days

    def apply(self, data: pd.Series, initial_investment: float) -> pd.Series:
        """
        策略实现逻辑
        :param data: 时间序列数据（资产价格）
        :param initial_investment: 初始投资金额
        :return: 每日资产份额的时间序列
        """
        # 计算长期均线
        long_ma = data.rolling(window=self.long_window).mean()

        # 初始化变量
        shares = 0  # 当前持仓（份额）
        cash = initial_investment  # 初始现金
        daily_target_shares = []  # 每日持仓记录
        last_signal = None  # 上一次的交易信号（"buy" 或 "sell"）

        # 遍历价格数据，逐日计算
        for i in range(len(data)):
            if i < self.long_window:  # 均线未计算完成时，跳过
                daily_target_shares.append(shares)
                continue

            # 当前价格与长期均线
            current_price = data.iloc[i]
            current_ma = long_ma.iloc[i]

            # 判断趋势信号
            if current_price > current_ma and all(data.iloc[i - self.confirmation_days:i] > current_ma):
                signal = "buy"  # 确认上涨趋势
            elif current_price < current_ma and all(data.iloc[i - self.confirmation_days:i] < current_ma):
                signal = "sell"  # 确认下跌趋势
            else:
                signal = None  # 无交易信号

            # 执行交易逻辑
            if signal == "buy" and last_signal != "buy":
                # 买入操作：全仓买入
                shares = cash / current_price
                cash = 0  # 用完所有现金
                last_signal = "buy"
            elif signal == "sell" and last_signal != "sell":
                # 卖出操作：清仓
                cash = shares * current_price
                shares = 0  # 清空持仓
                last_signal = "sell"

            # 记录当天的持仓
            daily_target_shares.append(shares)

        # 返回每日持仓份额的时间序列
        return pd.Series(daily_target_shares, index=data.index)


class BacktestSystem:
    """
    通用回测系统，支持动态策略类。
    """

    @staticmethod
    def backtest(
            data: pd.Series,
            strategy: BaseInvestmentStrategy,
            start_date=None,
            end_date=None,
            **kwargs
    ) -> (pd.DataFrame, List[Dict]):
        """
        回测系统，计算投资结果
        :param data: 时间序列数据（资产价格）
        :param strategy: 具体的投资策略类实例
        :param start_date: 回测起始日期（可选，str 或 datetime）
        :param end_date: 回测结束日期（可选，str 或 datetime）
        :return:  回测结果 DataFrame 和交易记录 trade_log
        """

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
        daily_target_shares = strategy.apply(data, **kwargs)

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


# === 可视化模块 ===
class Visualization:
    """
    可视化工具
    """

    @staticmethod
    def _setup_base_plot(title: str, xlabel: str = "Date", ylabel: str = "Value") -> Tuple[plt.Figure, plt.Axes]:
        """
        设置基础图表样式
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
        绘制价格和基准线对比图
        :param prices: 价格数据
        :param benchmark: 基准数据
        :param title: 图表标题
        :param price_label: 价格线标签
        :param benchmark_label: 基准线标签
        :return: matplotlib图表对象
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
        绘制回测结果
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
        绘制交易记录
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


# === 投资管理器 ===
class InvestmentManager:
    """
    投资管理器,负责协调数据加载和回测执行
    """

    def __init__(self):
        self.symbols = ["518800.SS", "^IXIC", "BTC-USD"]
        self.names = ["Gold", "NASDAQ", "BTC"]
        self.time_ranges = list(TIME_RANGES.keys())

        # 初始化数据
        self.data_loader = DataLoader(self.symbols)
        self._load_data()

    def _load_data(self):
        """加载并处理数据"""
        self.raw_data = self.data_loader.fetch_data().ffill()
        self.close_prices = self.raw_data.loc[:, (slice(None), 'Close')].dropna()
        self.close_prices = self.close_prices[self.symbols]
        self.close_prices.columns = self.names

    def get_asset_data(self, asset_name: str, time_range: str = "全部") -> pd.Series:
        """
        获取单个资产的价格数据
        """
        prices = self.close_prices[asset_name]

        if time_range != "全部":
            start_date = pd.Timestamp.now() - TIME_RANGES[time_range]
            prices = prices[prices.index >= start_date]

        return prices

    def execute_backtest(self,
                         asset_name: str,
                         strategy_type: str,
                         initial_investment: float,
                         start_date: str = None,
                         end_date: str = None) -> Tuple[pd.DataFrame, List[Dict], Dict]:
        """
        执行回测并返回结果
        :return: (回测结果, 交易记录, 回测指标)
        """
        # 获取资产数据
        asset_prices = self.get_asset_data(asset_name)

        # 创建策略实例
        if strategy_type == "Dollar Cost Averaging":
            strategy = DollarCostAveragingStrategy()
        else:  # Trend Following
            strategy = TrendFollowingStrategy()

        kwargs = {"initial_investment": initial_investment}

        # 执行回测
        results, trade_log = BacktestSystem.backtest(
            asset_prices, strategy,
            start_date=start_date,
            end_date=end_date,
            **kwargs
        )

        # 计算回测指标
        max_drawdown_info = Metrics.calculate_max_drawdown(results['Net Worth'])

        metrics = {
            'max_drawdown': max_drawdown_info,
        }

        return results, trade_log, metrics


def create_gradio_interface():
    manager = InvestmentManager()

    def format_metrics(results: pd.DataFrame, metrics: Dict) -> str:
        """格式化关键指标显示"""
        return f"""
### 投资概览
- **投资总额**: ${results['Total Investment'].iloc[-1]:,.2f}
- **最终净值**: ${results['Net Worth'].iloc[-1]:,.2f}
- **总收益**: ${results['Profit'].iloc[-1]:,.2f}
- **收益率**: {(results['Profit'].iloc[-1] / results['Total Investment'].iloc[-1] * 100):.2f}%
- **最大回撤**: {metrics['max_drawdown']['max_drawdown']:.2%}
"""

    def format_trade_log(trade_log: List[Dict]) -> str:
        """格式化交易记录显示"""
        if not trade_log:
            return "### 交易记录\n无交易记录"

        header = "### 交易记录\n\n| 日期 | 操作 | 份额 | 价格 | 金额 |"
        separator = "|:---:|:---:|---:|---:|---:|"
        rows = []

        for log in trade_log:
            date = log['date'].strftime('%Y-%m-%d')
            action = "买入" if log['action'] == 'buy' else "卖出"
            action_color = "🟢" if log['action'] == 'buy' else "🔴"
            rows.append(
                f"| {date} | {action_color}{action} | {log['shares']:.2f} | "
                f"${log['price']:.2f} | ${log['amount']:,.2f} |"
            )

        return "\n".join([header, separator] + rows)

    def validate_dates(start: str, end: str) -> Tuple[str, str]:
        """
        验证并调整日期范围
        """
        min_date = datetime(2015, 1, 1)
        max_date = datetime.now() - pd.Timedelta(days=1)  # 昨天

        # 转换输入日期为datetime对象
        start_date = pd.to_datetime(start) if start else min_date
        end_date = pd.to_datetime(end) if end else max_date

        # 检查并调整日期范围
        if start_date < min_date:
            start_date = min_date
        if end_date > max_date:
            end_date = max_date

        return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

    def analyze_asset(asset_name: str, time_range: str) -> plt.Figure:
        """资产分析"""
        prices = manager.get_asset_data(asset_name, time_range)

        # 计算通胀基准线
        inflation_benchmark = Metrics.calculate_inflation_benchmark(
            base_value=prices.iloc[0],
            dates=prices.index
        )

        return Visualization.plot_price_with_benchmark(
            prices=prices,
            benchmark=inflation_benchmark,
            title=asset_name,
            price_label=asset_name,
            benchmark_label="Inflation (7% Annual)"
        )

    def run_backtest_analysis(asset_name: str,
                              strategy_type: str,
                              initial_investment: float,
                              start_date: str = None,
                              end_date: str = None) -> Tuple[plt.Figure, plt.Figure, str]:
        """回测分析"""
        # 验证并调整日期范围
        start_date, end_date = validate_dates(start_date, end_date)

        # 执行回测
        results, trade_log, metrics = manager.execute_backtest(
            asset_name, strategy_type,
            initial_investment,
            start_date, end_date
        )

        # 生成图表
        backtest_fig = Visualization.plot_backtest_results(
            results, metrics['max_drawdown'],
            f"Backtest Results for {asset_name}"
        )
        trades_fig = Visualization.plot_trades(
            manager.get_asset_data(asset_name),
            trade_log,
            f"Trades for {asset_name}",
            start_date, end_date
        )

        # 生成指标报告
        metrics_text = format_metrics(results, metrics)
        trade_log_text = format_trade_log(trade_log)

        return backtest_fig, trades_fig, metrics_text, trade_log_text

    # 预先生成默认资产(Gold)的图表
    default_plot = analyze_asset(manager.names[0], "全部")

    with gr.Blocks(theme=gr.themes.Soft()) as interface:

        with gr.Tabs():
            # 价格标签页
            with gr.Tab("价格"):
                with gr.Row():
                    with gr.Column(scale=2):
                        asset_selector = gr.Dropdown(
                            choices=manager.names,
                            label="投资对象",
                            value=manager.names[0]
                        )
                    with gr.Column(scale=3):
                        time_range_selector = gr.Radio(
                            choices=manager.time_ranges,
                            label="时间范围",
                            value="全部",
                            container=False
                        )

                price_plot = gr.Plot(default_plot)

            # 回测标签页
            with gr.Tab("回测"):
                with gr.Row():
                    with gr.Column(scale=1):
                        backtest_asset = gr.Dropdown(
                            choices=manager.names,
                            label="投资对象",
                            value=manager.names[0]
                        )
                    with gr.Column(scale=1):
                        strategy = gr.Dropdown(
                            choices=["Dollar Cost Averaging", "Trend Following"],
                            label="投资策略",
                            value="Dollar Cost Averaging"
                        )

                with gr.Row():
                    with gr.Column(scale=1):
                        start_date = gr.DateTime(
                            label="开始日期",
                            include_time=False,
                            type="string",
                        )
                    with gr.Column(scale=1):
                        end_date = gr.DateTime(
                            label="结束日期",
                            include_time=False,
                            type="string",
                        )
                    with gr.Column(scale=1):
                        initial_inv = gr.Number(
                            label="初始投资额",
                            value=30000
                        )

                run_btn = gr.Button("运行回测")

                with gr.Row():
                    backtest_plot = gr.Plot(label="回测结果")
                    trades_plot = gr.Plot(label="交易明细")

                with gr.Row():
                    with gr.Column(scale=1):
                        metrics_output = gr.Markdown(
                            label="关键指标"
                        )
                    with gr.Column(scale=1):
                        trade_log_output = gr.Markdown(
                            label="交易记录"
                        )

                asset_selector.change(
                    analyze_asset,
                    inputs=[asset_selector, time_range_selector],
                    outputs=price_plot
                )
                time_range_selector.change(
                    analyze_asset,
                    inputs=[asset_selector, time_range_selector],
                    outputs=price_plot
                )
                run_btn.click(
                    run_backtest_analysis,
                    inputs=[
                        backtest_asset,
                        strategy,
                        initial_inv,
                        start_date,
                        end_date
                    ],
                    outputs=[
                        backtest_plot,
                        trades_plot,
                        metrics_output,
                        trade_log_output
                    ]
                )

    return interface


if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch()
import gradio as gr
import pandas as pd
import yaml
from typing import Tuple, List, Dict

import matplotlib.pyplot as plt
from datetime import datetime

from automated_trading.data.loaders import DataLoader
from automated_trading.strategies.registry import StrategyRegistry
from automated_trading.backtest.backtest import BacktestSystem
from automated_trading.metrics.calculations import Metrics
from automated_trading.visualization.plots import Visualization

# 常量定义
TIME_RANGES = {
    "全部": None,
    "5年": pd.DateOffset(years=5),
    "1年": pd.DateOffset(years=1),
    "6个月": pd.DateOffset(months=6),
    "1个月": pd.DateOffset(months=1)
}


class InvestmentManager:
    """
    投资管理器,负责协调数据加载和回测执行。
    """

    def __init__(self):
        self.symbols = ["518800.SS", "^IXIC", "BTC-USD"]
        self.names = ["Gold", "NASDAQ", "BTC"]
        self.time_ranges = list(TIME_RANGES.keys())
        self.strategy_registry = StrategyRegistry()

        # 初始化数据
        self.data_loader = DataLoader(self.symbols)
        self._load_data()

    def _load_data(self):
        """加载并处理数据。"""
        self.raw_data = self.data_loader.fetch_data().ffill()
        self.close_prices = self.raw_data.loc[:, (slice(None), 'Close')].dropna()
        self.close_prices = self.close_prices[self.symbols]
        self.close_prices.columns = self.names

    def get_asset_data(self, asset_name: str, time_range: str = "全部") -> pd.Series:
        """
        获取单个资产的价格数据。
        """
        prices = self.close_prices[asset_name]

        if time_range != "全部":
            start_date = pd.Timestamp.now() - TIME_RANGES[time_range]
            prices = prices[prices.index >= start_date]

        return prices

    def execute_backtest(self,
                         asset_name: str,
                         strategy_name: str,
                         start_date: str = None,
                         end_date: str = None,
                         **strategy_params) -> Tuple[pd.DataFrame, List[Dict], Dict]:
        """
        执行回测并返回结果。
        """
        # 获取资产数据
        asset_prices = self.get_asset_data(asset_name)

        # 创建策略实例
        strategy = self.strategy_registry.create_strategy(strategy_name)

        # 执行回测
        results, trade_log = BacktestSystem.backtest(
            asset_prices, strategy,
            start_date=start_date,
            end_date=end_date,
            **strategy_params
        )

        # 计算回测指标
        max_drawdown_info = Metrics.calculate_max_drawdown(results['Net Worth'])

        metrics = {
            'max_drawdown': max_drawdown_info,
        }

        return results, trade_log, metrics


def create_gradio_interface():
    manager = InvestmentManager()

    def update_strategy_info(strategy_name: str) -> str:
        """
        更新策略说明。
        """
        strategy = manager.strategy_registry.create_strategy(strategy_name)
        return strategy.get_strategy_info()

    def parse_strategy_params(yaml_text: str) -> dict:
        """
        解析策略参数并验证参数类型。
        返回：参数字典
        """
        if not yaml_text or yaml_text.strip() == "":
            return {}

        # 尝试解析为YAML格式
        try:
            params = yaml.safe_load(yaml_text)
            if isinstance(params, dict):
                return params
        except yaml.YAMLError:
            pass

        # 尝试解析为键值对格式
        try:
            params = {}
            for line in yaml_text.strip().split('\n'):
                line = line.strip()
                if line and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    try:
                        if value.isdigit():
                            value = int(value)
                        else:
                            value = float(value)
                    except ValueError:
                        pass
                    params[key] = value
            return params
        except Exception:
            return {}

    def format_metrics(results: pd.DataFrame, metrics: Dict) -> str:
        """格式化关键指标显示。"""
        return f"""
### 投资概览
- **投资总额**: ${results['Total Investment'].iloc[-1]:,.2f}
- **最终净值**: ${results['Net Worth'].iloc[-1]:,.2f}
- **总收益**: ${results['Profit'].iloc[-1]:,.2f}
- **收益率**: {(results['Profit'].iloc[-1] / results['Total Investment'].iloc[-1] * 100):.2f}%
- **最大回撤**: {metrics['max_drawdown']['max_drawdown']:.2%}
"""

    def format_trade_log(trade_log: List[Dict]) -> str:
        """格式化交易记录显示。"""
        if not trade_log:
            return "### 交易记录\n无交易记录。"

        header = "### 交易记录\n\n| 日期 | 操作 | 份额 | 价格 | 金额 |"
        separator = "|:---:|:---:|---:|---:|---:|"
        rows = []

        for log in trade_log:
            date = log['date'].strftime('%Y-%m-%d')
            action = "买入" if log['action'] == 'buy' else "卖出"
            action_icon = "🟢" if log['action'] == 'buy' else "🔴"
            rows.append(
                f"| {date} | {action_icon}{action} | {log['shares']:.2f} | "
                f"${log['price']:.2f} | ${log['amount']:,.2f} |"
            )

        return "\n".join([header, separator] + rows)

    def validate_dates(start: str, end: str) -> Tuple[str, str]:
        """
        验证并调整日期范围。
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
        """资产分析。"""
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
                              strategy_params: str = None,
                              start_date: str = None,
                              end_date: str = None) -> Tuple[plt.Figure, plt.Figure, str, str]:
        """回测分析。"""
        try:
            # 检查必要参数
            if asset_name is None:
                raise ValueError("### 请选择投资对象。")
            if strategy_type is None:
                raise ValueError("### 请选择投资策略。")

            # 验证并调整日期范围
            start_date, end_date = validate_dates(start_date, end_date)

            # 解析策略参数
            params = parse_strategy_params(strategy_params)
            params['initial_investment'] = initial_investment

            # 执行回测
            results, trade_log, metrics = manager.execute_backtest(
                asset_name, strategy_type,
                start_date=start_date,
                end_date=end_date,
                **params
            )

            # 生成图表和报告
            backtest_fig = Visualization.plot_backtest_results(
                results, metrics['max_drawdown'],
                f"Backtest Results for - {asset_name}"
            )
            trades_fig = Visualization.plot_trades(
                manager.get_asset_data(asset_name),
                trade_log,
                f"Trades for - {asset_name}",
                start_date, end_date
            )

            metrics_text = format_metrics(results, metrics)
            trade_log_text = format_trade_log(trade_log)

            return backtest_fig, trades_fig, metrics_text, trade_log_text

        except ValueError as e:
            return None, None, f"{str(e)}", ""

    # 预先生成默认资产（Gold）的图表
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
                            value=None
                        )
                    with gr.Column(scale=1):
                        strategy = gr.Dropdown(
                            choices=StrategyRegistry.get_strategy_names(),
                            label="投资策略",
                            value=None
                        )

                with gr.Row():
                    with gr.Column(scale=1):
                        start_date = gr.DateTime(
                            label="开始日期（可选）",
                            include_time=False,
                            type="string",
                        )
                    with gr.Column(scale=1):
                        end_date = gr.DateTime(
                            label="结束日期（可选）",
                            include_time=False,
                            type="string",
                        )
                    with gr.Column(scale=1):
                        initial_inv = gr.Number(
                            label="初始投资额",
                            value=10000,
                            minimum=1
                        )

                with gr.Row():
                    with gr.Column(scale=1):
                        strategy_params = gr.Textbox(
                            label="策略参数（可选）",
                            placeholder=(
                                "示例:\n"
                                "long_window=200\n"
                                "confirmation_days=10\n"
                                "或:\n"
                                "long_window: 200\n"
                                "confirmation_days: 10"
                            ),
                            lines=5
                        )
                    with gr.Column(scale=1):
                        strategy_info = gr.Markdown(
                            label="策略说明"
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

                # 事件绑定
                strategy.change(
                    update_strategy_info,
                    inputs=[strategy],
                    outputs=[strategy_info]
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
                        strategy_params,
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

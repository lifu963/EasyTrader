import re
from typing import List, Type
from .base import BaseInvestmentStrategy


class StrategyRegistry:
    """
    策略注册器 - 单例模式
    """
    _instance = None
    _strategies = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(StrategyRegistry, cls).__new__(cls)
        return cls._instance

    @classmethod
    def register(cls, strategy_class: Type[BaseInvestmentStrategy]):
        """
        注册策略。
        """
        if not issubclass(strategy_class, BaseInvestmentStrategy):
            raise TypeError(f"{strategy_class.__name__} 必须继承 BaseInvestmentStrategy")

        # 使用策略类的名称作为键，去除 'Strategy' 后缀
        strategy_name = strategy_class.__name__.replace('Strategy', '')
        # 转换为更友好的显示名称（例如：DollarCostAveraging -> Dollar Cost Averaging）
        display_name = ' '.join(re.findall('[A-Z][^A-Z]*', strategy_name))
        cls._strategies[display_name] = strategy_class

        return strategy_class  # 返回策略类以支持装饰器用法

    @classmethod
    def get_strategy(cls, strategy_name: str) -> Type[BaseInvestmentStrategy]:
        """
        获取策略类。
        """
        return cls._strategies.get(strategy_name)

    @classmethod
    def get_strategy_names(cls) -> List[str]:
        """
        获取所有已注册的策略名称。
        """
        return list(cls._strategies.keys())

    @classmethod
    def create_strategy(cls, strategy_name: str) -> BaseInvestmentStrategy:
        """
        创建策略实例。
        """
        strategy_class = cls.get_strategy(strategy_name)
        if strategy_class is None:
            raise ValueError(f"未知的策略: {strategy_name}")
        return strategy_class()
    
# EasyTrader

> 一个极简的、无脑的自动交易程序，专为不想天天盯盘的懒人设计 🚀

## 🎯 核心理念

- 普通人理财的目标是保值，而非暴利
- 越是长周期、大周期，就越容易被预测、越值得被押注
- 锚定优质资产（如美股、黄金等）
- 不追涨不杀跌，顺应大趋势
- 目标是稳健保值，跟上时代平均水平

## 💡 投资策略

核心策略：不追涨，不杀跌。在大趋势形成后再行动：
- 当趋势向上时买入
- 当趋势回落时卖出

旨在为投资者提供一个安全、简单且可靠的交易方案。

## ✨ 主要特性

- 开箱即用，无需复杂配置
- 直观的数据可视化展示
- 灵活的策略自定义能力
- 支持 Docker 一键部署

## 🚀 快速开始

### 方式一：Python

```bash
# python >= 3.11
pip install -r requirements.txt
cd automated_trading/
python main.py
```

### 方式二：Docker

```bash
docker pull lifu963/easy-trader:v2.0
docker run lifu963/easy-trader:v2.0
```

访问 localhost:7890 即可开始使用

## 📊 运行效果

### 资产价格展示

<img width="884" alt="8f2edaaad72065e8e10595510eae627" src="https://github.com/user-attachments/assets/4e8f7665-cf22-4ec3-81bf-2335f1137f75" />

### 回测投资策略

<img width="884" alt="e1ba50bee4a7fc8624abf0c1e6168e0" src="https://github.com/user-attachments/assets/5782243e-64fe-452c-926d-52a90f0fa2d2" />

### 回测结果展示

<img width="855" alt="c599bc289b3dd61e3a25e723be43d51" src="https://github.com/user-attachments/assets/d892ef63-f6c3-4746-9a78-52663cb420af" />

## 🔧 自定义交易策略

只需三步即可添加新策略：

1. 在 `strategies/` 目录下创建新策略类
2. 继承 BaseInvestmentStrategy，实现 apply 方法
3. 在 `strategies/__init__.py` 中导入

```python
@StrategyRegistry.register
class TrendFollowingStrategy(BaseInvestmentStrategy):
    def apply(self, data: pd.Series, **kwargs) -> pd.Series:
        # 输入：股票每日收盘价的时间序列
        # 策略逻辑实现
        # 返回：每日持仓份额的时间序列
```

详细参考请查看 strategies/ 目录下的示例策略。

## ⚠️ 已知问题

- 首次启动需要通过 yfinance 下载数据，需要科学上网

## 📝 开发计划

- [ ] 扩充优质资产池
- [ ] 替换数据源，消除网络限制
- [ ] 增加策略评价指标
- [ ] 完善趋势跟踪策略
- [ ] 集成新闻分析功能

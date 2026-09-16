import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class PortfolioStrategy:
    def __init__(self, num_assets=16, start_date='2011-12-31', end_date='2023-12-31', 
                 risk_budget=0.1, top_n=12, vol_threshold=0.06, rebalance_days=[4, 5, 6, 7]):
        self.num_assets = num_assets
        self.virtual_start_date = '2009-12-31'  # 虚拟数据的开始日期
        self.start_date = start_date  # 实际计算收益的开始日期
        self.end_date = end_date
        self.risk_budget = risk_budget
        self.top_n = top_n
        self.vol_threshold = vol_threshold
        self.rebalance_days = rebalance_days

        # 定义资产名称及其最大持仓比例
        self.assets_limits = {
            '沪深300': 0.10,
            '中证500': 0.10,
            '日经225': 0.10,
            '标普500': 0.15,
            '德国DAX': 0.10,
            '中国5年国债': 0.40,
            '中国10年国债': 0.40,
            '美国2年国债': 0.60,
            '美国10年国债': 0.40,
            '德国长期国债': 0.40,
            '日本长期国债': 0.40,
            'Brent原油': 0.10,
            '沪金': 0.10,
            '沪铜': 0.05,
            '豆粕': 0.05,
            '黑色系': 0.10
        }

        self.price_data = None
        self.returns = None
        self.signals = None
        self.weights = None
        self.metrics = None

    def generate_virtual_data(self):
        workdays = pd.bdate_range(start=self.virtual_start_date, end='2023-12-31')
        num_days = len(workdays)  # 工作日的数量
        dates = pd.date_range(start=self.virtual_start_date, periods=num_days, freq='B')  # 只包含工作日
        assets = list(self.assets_limits.keys())
        
        np.random.seed(42)  # 设置随机种子以便重现
        self.price_data = pd.DataFrame(index=dates)

        for asset in assets:
            self.price_data[asset] = 100 * (1 + np.random.normal(0, 0.01, num_days)).cumprod()

        self.price_data.to_csv('virtual_asset_prices.csv')

    def calculate_returns(self):
        self.returns = self.price_data.pct_change().dropna()
        self.returns = self.returns[self.returns.index >= self.virtual_start_date]

    def calculate_signals(self):
    # 创建一个空的 DataFrame 用于存储信号
        signals = pd.DataFrame(index=self.returns.index)

    # 设定窗口大小
        one_month_window = 21
        three_month_window = 63
        six_month_window = 126

    # 针对每个资产单独计算信号
        for asset in self.returns.columns:
        # 计算 1M 信号
            signals.loc[:, f'{asset}_1M'] = self.returns[asset].rolling(window=one_month_window).sum()

        # 计算 3M/6M_Vol 信号
            sum_3m = self.returns[asset].rolling(window=three_month_window).sum()
            std_6m = self.returns[asset].rolling(window=six_month_window).std()
            signals.loc[:, f'{asset}_3M/6M_Vol'] = sum_3m / std_6m

        # 计算 6M/6M_Vol 信号
            sum_6m = self.returns[asset].rolling(window=six_month_window).sum()
            signals.loc[:, f'{asset}_6M/6M_Vol'] = sum_6m / std_6m

        # 计算 Mean_6M/6M_Vol 信号
            mean_6m = self.returns[asset].rolling(window=six_month_window).mean()
            signals.loc[:, f'{asset}_Mean_6M/6M_Vol'] = mean_6m / std_6m

        self.signals = signals

    def select_top_assets(self):
    # 计算每个资产的四个信号的平均值
        average_signals = self.signals.mean(axis=0)

    # 提取每个资产的信号
        asset_averages = {}
        for asset in self.returns.columns:
            asset_signals = [average_signals[f'{asset}_1M'], average_signals[f'{asset}_3M/6M_Vol'],
                             average_signals[f'{asset}_6M/6M_Vol'], average_signals[f'{asset}_Mean_6M/6M_Vol']]
            asset_averages[asset] = np.mean(asset_signals)

    # 选择平均值最大的前 top_n 个资产
        top_assets = sorted(asset_averages, key=asset_averages.get, reverse=True)[:self.top_n]
        return top_assets


    def allocate_weights(self, top_assets):
        weights = pd.Series(index=top_assets, dtype=float)
        risk_budget_per_asset = self.risk_budget / self.top_n

        # 记录高波动资产
        high_vol_assets = []
        adjusted_weights = {}
        remain_assets = []

        # 计算高波动资产权重
        for asset in top_assets:
            if asset in self.returns.columns:  # 确保资产在收益数据中
                historical_volatility_22 = self.returns[asset].rolling(window=22).std().iloc[-1]
                historical_volatility_65 = self.returns[asset].rolling(window=65).std().iloc[-1]
                historical_volatility_130 = self.returns[asset].rolling(window=130).std().iloc[-1]
                max_volatility = max(historical_volatility_22, historical_volatility_65, historical_volatility_130)

                # 检查波动率是否超标
                if max_volatility > self.vol_threshold:
                    high_vol_assets.append(asset)  # 记录高波动资产
                    adjusted_weights[asset] = 0.05  # 高波动资产权重设定为5%
                else:
                    # 计算初始权重
                    adjusted_weights[asset] = risk_budget_per_asset / max_volatility
            else:
                pass

        # 计算高波动资产的总权重
        total_high_vol_weight = len(high_vol_assets) * 0.05
        remaining_weight = 1.0 - total_high_vol_weight

        # 计算非高波动资产
        non_high_vol_assets = [asset for asset in adjusted_weights.keys() if asset not in high_vol_assets]

        if non_high_vol_assets:
            # 计算非高波动资产的总初步权重
            total_non_high_vol_weight = sum(adjusted_weights[asset] for asset in non_high_vol_assets)

            # 按比例分配剩余权重给其他资产
            for asset in non_high_vol_assets:
                adjusted_weights[asset] = (adjusted_weights[asset] / total_non_high_vol_weight) * remaining_weight

        total_weight = remaining_weight

        # 处理超过最大持仓比例的资产
        for asset in adjusted_weights.keys():
            if adjusted_weights[asset] > self.assets_limits[asset]:
                remaining_weight -= adjusted_weights[asset]
                adjusted_weights[asset] = self.assets_limits[asset]
                total_weight -= adjusted_weights[asset] 
            else:
                remain_assets.append(asset)

        common_assets = set(remain_assets) & set(non_high_vol_assets)
        for asset in common_assets:
            adjusted_weights[asset] = (adjusted_weights[asset] * total_weight) / remaining_weight

        self.weights = pd.Series(adjusted_weights)
    
    def calculate_metrics(self):
        portfolio_returns = (self.weights * self.returns).sum(axis=1)

        annual_returns = portfolio_returns.resample('YE').apply(lambda x: (1 + x).prod() - 1)  # 使用 'Y'
        annual_volatility = portfolio_returns.resample('YE').std() * np.sqrt(252)
    
        # 避免除以零
        sharpe_ratio = annual_returns / annual_volatility.replace(0, np.nan)  # 替换0为NaN以避免除以零
        max_drawdown = (portfolio_returns.cumsum() - portfolio_returns.cumsum().cummax()).min()

        self.metrics = pd.DataFrame({
            'Annual Returns': annual_returns,
            'Annual Volatility': annual_volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown
        })
        self.metrics.to_csv('annual_metrics.csv')

    def plot_net_value(self):
        portfolio_returns = (self.weights * self.returns).sum(axis=1)
        net_value = (1 + portfolio_returns).cumprod()

        plt.figure(figsize=(10, 6))
        net_value.plot(title='Portfolio Net Value (Starting at 1)')
        plt.xlabel('Year')
        plt.ylabel('Net Value')
        plt.axhline(1, color='red', linestyle='--')  # 初始值为1
        plt.grid()
        plt.savefig('portfolio_net_value.png')
        plt.show()

    def rebalance(self):
        trading_days = self.returns.index.date
        for day in self.rebalance_days:
            if day in trading_days:
                self.allocate_weights(self.select_top_assets())

    def run(self):
        self.generate_virtual_data()
        self.calculate_returns()
        self.calculate_signals()  # 计算信号
        self.rebalance()  # 调仓
        top_assets = self.select_top_assets()
        self.allocate_weights(top_assets)
        self.calculate_metrics()
        self.plot_net_value()

# 使用示例
if __name__ == "__main__":
    strategy = PortfolioStrategy()
    strategy.run()
        "Japan Long Treasury": 0.40,
        "Brent Oil": 0.10,
        "SHFE Gold": 0.10,
        "SHFE Copper": 0.05,
        "Soybean Meal": 0.05,
        "Ferrous Metals": 0.10,
    },
)


GMAT3_STYLE = StrategyConfig(
    name="gmat3",
    top_n=11,
    vol_threshold=0.045,
    asset_caps={
        "CSI 300": 0.10,
        "CSI 500": 0.10,
        "CSI 1000": 0.10,
        "Nasdaq 100": 0.10,
        "S&P 500": 0.10,
        "China 2Y Treasury": 0.40,
        "China 5Y Treasury": 0.40,
        "China 10Y Treasury": 0.40,
        "US 2Y Treasury": 0.60,
        "US 5Y Treasury": 0.40,
        "US 10Y Treasury": 0.40,
        "Brent Oil": 0.10,
        "SHFE Gold": 0.10,
        "SHFE Copper": 0.10,
        "Soybean Meal": 0.10,
        "Ferrous Metals": 0.10,
    },
)


class PortfolioStrategy:
    def __init__(
        self,
        config: StrategyConfig,
        start_date: str = "2011-12-31",
        end_date: str = "2023-12-31",
        risk_budget: float = 0.10,
        seed: int = 42,
    ) -> None:
        self.config = config
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.virtual_start_date = pd.Timestamp("2009-12-31")
        self.risk_budget = risk_budget
        self.seed = seed

    def generate_virtual_prices(self) -> pd.DataFrame:
        dates = pd.bdate_range(start=self.virtual_start_date, end=self.end_date)
        rng = np.random.default_rng(self.seed)
        prices = pd.DataFrame(index=dates)

        for asset in self.config.asset_caps:
            shocks = rng.normal(loc=0.00015, scale=0.01, size=len(dates))
            prices[asset] = 100 * np.cumprod(1 + shocks)

        return prices

    @staticmethod
    def _mass260(price: pd.Series) -> pd.Series:
        averages = [price.rolling(window=window).mean() for window in range(1, 261)]
        scores = [(averages[i] >= averages[i + 1]).astype(float) for i in range(len(averages) - 1)]
        return pd.concat(scores, axis=1).mean(axis=1)

    def calculate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        returns = prices.pct_change()
        signals = pd.DataFrame(index=returns.index)

        for asset in returns.columns:
            if self.config.name == "gmat2":
                ret_1m = returns[asset].rolling(21).sum()
                ret_3m = returns[asset].rolling(63).sum()
                ret_6m = returns[asset].rolling(126).sum()
                vol_6m = returns[asset].rolling(126).std()
                signals[asset] = pd.concat(
                    [
                        ret_1m,
                        ret_3m / vol_6m,
                        ret_6m / vol_6m,
                        returns[asset].rolling(126).mean() / vol_6m,
                    ],
                    axis=1,
                ).mean(axis=1)
            else:
                ret_1m = returns[asset].rolling(21).sum()
                vol_1m = returns[asset].rolling(21).std()
                ret_12m = returns[asset].rolling(252).sum()
                vol_12m = returns[asset].rolling(252).std()
                low_12m = prices[asset].rolling(252).min()
                high_12m = prices[asset].rolling(252).max()
                price_position = (prices[asset] - low_12m) / (high_12m - low_12m)
                mass260 = self._mass260(prices[asset])
                signals[asset] = pd.concat(
                    [ret_1m / vol_1m, ret_12m / vol_12m, ret_12m, price_position, mass260],
                    axis=1,
                ).mean(axis=1)

        return signals

    def allocate(self, returns: pd.DataFrame, signals: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
        eligible = signals.loc[date].dropna().nlargest(self.config.top_n).index
        weights = pd.Series(0.0, index=returns.columns)
        if len(eligible) == 0:
            return weights

        recent_vol = returns.loc[:date, eligible].tail(130).std()
        recent_vol = recent_vol.replace(0, np.nan).dropna()
        if recent_vol.empty:
            return weights

        raw = (self.risk_budget / self.config.top_n) / recent_vol
        high_vol = recent_vol > self.config.vol_threshold
        raw.loc[high_vol] = 0.04

        for asset, cap in self.config.asset_caps.items():
            if asset in raw.index:
                raw.loc[asset] = min(raw.loc[asset], cap)

        if raw.sum() > 0:
            weights.loc[raw.index] = raw / raw.sum()

        return weights

    def backtest(self, prices: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
        returns = prices.pct_change().dropna()
        signals = self.calculate_signals(prices)
        rebalancing_dates = returns.loc[self.start_date : self.end_date].resample("ME").last().index

        weights = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
        current_weights = pd.Series(0.0, index=returns.columns)

        for date in returns.index:
            if date in rebalancing_dates:
                current_weights = self.allocate(returns, signals, date)
            weights.loc[date] = current_weights

        strategy_returns = (weights.shift(1).fillna(0) * returns).sum(axis=1)
        strategy_returns = strategy_returns.loc[self.start_date : self.end_date]
        net_value = (1 + strategy_returns).cumprod()
        metrics = self.calculate_metrics(strategy_returns)
        return net_value, metrics, weights.loc[self.start_date : self.end_date]

    @staticmethod
    def calculate_metrics(strategy_returns: pd.Series) -> pd.DataFrame:
        annual_returns = strategy_returns.resample("YE").apply(lambda x: (1 + x).prod() - 1)
        annual_volatility = strategy_returns.resample("YE").std() * np.sqrt(252)
        sharpe = annual_returns / annual_volatility.replace(0, np.nan)
        cumulative = (1 + strategy_returns).cumprod()
        drawdown = cumulative / cumulative.cummax() - 1

        return pd.DataFrame(
            {
                "annual_return": annual_returns,
                "annual_volatility": annual_volatility,
                "sharpe_ratio": sharpe,
                "max_drawdown_to_date": drawdown.resample("YE").min(),
            }
        )

    @staticmethod
    def plot_net_value(net_value: pd.Series, output_path: Path) -> None:
        fig, ax = plt.subplots(figsize=(10, 6))
        net_value.plot(ax=ax, color="#0f766e", linewidth=2)
        ax.axhline(1.0, color="#dc2626", linestyle="--", linewidth=1)
        ax.set_title("Portfolio Net Value")
        ax.set_xlabel("Date")
        ax.set_ylabel("Net value")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a GMAT-style multi-asset allocation prototype.")
    parser.add_argument("--version", choices=["gmat2", "gmat3"], default="gmat3")
    parser.add_argument("--output-dir", type=Path, default=Path("results/task2"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = GMAT2_STYLE if args.version == "gmat2" else GMAT3_STYLE
    strategy = PortfolioStrategy(config=config, seed=args.seed)
    prices = strategy.generate_virtual_prices()
    net_value, metrics, weights = strategy.backtest(prices)

    prices.to_csv(args.output_dir / f"virtual_asset_prices_{config.name}_style.csv")
    weights.to_csv(args.output_dir / f"weights_{config.name}_style.csv")
    metrics.to_csv(args.output_dir / f"annual_metrics_{config.name}_style.csv")
    strategy.plot_net_value(net_value, args.output_dir / f"portfolio_net_value_{config.name}_style.png")


if __name__ == "__main__":
    main()

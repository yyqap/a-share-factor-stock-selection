"""GMAT-style global multi-asset trend-following allocation prototype."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    top_n: int
    vol_threshold: float
    asset_caps: dict[str, float]


GMAT2_STYLE = StrategyConfig(
    name="gmat2",
    top_n=12,
    vol_threshold=0.06,
    asset_caps={
        "CSI 300": 0.10,
        "CSI 500": 0.10,
        "Nikkei 225": 0.10,
        "S&P 500": 0.15,
        "DAX": 0.10,
        "China 5Y Treasury": 0.40,
        "China 10Y Treasury": 0.40,
        "US 2Y Treasury": 0.60,
        "US 10Y Treasury": 0.40,
        "Germany Long Treasury": 0.40,
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

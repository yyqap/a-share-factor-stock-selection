import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


class PortfolioStrategy:
    def __init__(
        self,
        num_assets=16,
        start_date='2011-12-31',
        end_date='2023-12-31',
        risk_budget=0.1,
        top_n=12,
        vol_threshold=0.06,
        rebalance_days=[4, 5, 6, 7]
    ):
        self.num_assets = num_assets
        self.virtual_start_date = '2009-12-31'  # Start date for virtual data
        self.start_date = start_date  # Start date for actual return calculation
        self.end_date = end_date
        self.risk_budget = risk_budget
        self.top_n = top_n
        self.vol_threshold = vol_threshold
        self.rebalance_days = rebalance_days

        # Define asset names and their maximum position limits
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
        workdays = pd.bdate_range(
            start=self.virtual_start_date,
            end='2023-12-31'
        )

        num_days = len(workdays)  # Number of business days

        dates = pd.date_range(
            start=self.virtual_start_date,
            periods=num_days,
            freq='B'
        )  # Include business days only

        assets = list(self.assets_limits.keys())

        np.random.seed(42)  # Set random seed for reproducibility

        self.price_data = pd.DataFrame(index=dates)

        for asset in assets:
            self.price_data[asset] = (
                100
                * (
                    1
                    + np.random.normal(0, 0.01, num_days)
                ).cumprod()
            )

        self.price_data.to_csv('virtual_asset_prices.csv')

    def calculate_returns(self):
        self.returns = self.price_data.pct_change().dropna()

        self.returns = self.returns[
            self.returns.index >= self.virtual_start_date
        ]

    def calculate_signals(self):
        # Create an empty DataFrame to store signals
        signals = pd.DataFrame(index=self.returns.index)

        # Set window sizes
        one_month_window = 21
        three_month_window = 63
        six_month_window = 126

        # Calculate signals separately for each asset
        for asset in self.returns.columns:

            # Calculate the 1M signal
            signals.loc[:, f'{asset}_1M'] = (
                self.returns[asset]
                .rolling(window=one_month_window)
                .sum()
            )

            # Calculate the 3M/6M_Vol signal
            sum_3m = (
                self.returns[asset]
                .rolling(window=three_month_window)
                .sum()
            )

            std_6m = (
                self.returns[asset]
                .rolling(window=six_month_window)
                .std()
            )

            signals.loc[:, f'{asset}_3M/6M_Vol'] = (
                sum_3m / std_6m
            )

            # Calculate the 6M/6M_Vol signal
            sum_6m = (
                self.returns[asset]
                .rolling(window=six_month_window)
                .sum()
            )

            signals.loc[:, f'{asset}_6M/6M_Vol'] = (
                sum_6m / std_6m
            )

            # Calculate the Mean_6M/6M_Vol signal
            mean_6m = (
                self.returns[asset]
                .rolling(window=six_month_window)
                .mean()
            )

            signals.loc[:, f'{asset}_Mean_6M/6M_Vol'] = (
                mean_6m / std_6m
            )

        self.signals = signals

    def select_top_assets(self):
        # Calculate the average of the four signals for each asset
        average_signals = self.signals.mean(axis=0)

        # Extract signals for each asset
        asset_averages = {}

        for asset in self.returns.columns:
            asset_signals = [
                average_signals[f'{asset}_1M'],
                average_signals[f'{asset}_3M/6M_Vol'],
                average_signals[f'{asset}_6M/6M_Vol'],
                average_signals[f'{asset}_Mean_6M/6M_Vol']
            ]

            asset_averages[asset] = np.mean(asset_signals)

        # Select the top top_n assets with the highest average signals
        top_assets = sorted(
            asset_averages,
            key=asset_averages.get,
            reverse=True
        )[:self.top_n]

        return top_assets

    def allocate_weights(self, top_assets):
        weights = pd.Series(
            index=top_assets,
            dtype=float
        )

        risk_budget_per_asset = (
            self.risk_budget / self.top_n
        )

        # Record high-volatility assets
        high_vol_assets = []
        adjusted_weights = {}
        remain_assets = []

        # Calculate weights for high-volatility assets
        for asset in top_assets:

            if asset in self.returns.columns:  # Ensure the asset is in the return data

                historical_volatility_22 = (
                    self.returns[asset]
                    .rolling(window=22)
                    .std()
                    .iloc[-1]
                )

                historical_volatility_65 = (
                    self.returns[asset]
                    .rolling(window=65)
                    .std()
                    .iloc[-1]
                )

                historical_volatility_130 = (
                    self.returns[asset]
                    .rolling(window=130)
                    .std()
                    .iloc[-1]
                )

                max_volatility = max(
                    historical_volatility_22,
                    historical_volatility_65,
                    historical_volatility_130
                )

                # Check whether volatility exceeds the threshold
                if max_volatility > self.vol_threshold:

                    high_vol_assets.append(
                        asset
                    )  # Record high-volatility asset

                    adjusted_weights[asset] = (
                        0.05
                    )  # Set high-volatility asset weight to 5%

                else:
                    # Calculate the initial weight
                    adjusted_weights[asset] = (
                        risk_budget_per_asset
                        / max_volatility
                    )

            else:
                pass

        # Calculate the total weight of high-volatility assets
        total_high_vol_weight = (
            len(high_vol_assets) * 0.05
        )

        remaining_weight = (
            1.0 - total_high_vol_weight
        )

        # Calculate non-high-volatility assets
        non_high_vol_assets = [
            asset
            for asset in adjusted_weights.keys()
            if asset not in high_vol_assets
        ]

        if non_high_vol_assets:

            # Calculate the total initial weight of non-high-volatility assets
            total_non_high_vol_weight = sum(
                adjusted_weights[asset]
                for asset in non_high_vol_assets
            )

            # Allocate the remaining weight proportionally to other assets
            for asset in non_high_vol_assets:
                adjusted_weights[asset] = (
                    adjusted_weights[asset]
                    / total_non_high_vol_weight
                ) * remaining_weight

        total_weight = remaining_weight

        # Handle assets whose weights exceed their maximum position limits
        for asset in adjusted_weights.keys():

            if (
                adjusted_weights[asset]
                > self.assets_limits[asset]
            ):

                remaining_weight -= (
                    adjusted_weights[asset]
                )

                adjusted_weights[asset] = (
                    self.assets_limits[asset]
                )

                total_weight -= (
                    adjusted_weights[asset]
                )

            else:
                remain_assets.append(asset)

        common_assets = (
            set(remain_assets)
            & set(non_high_vol_assets)
        )

        for asset in common_assets:
            adjusted_weights[asset] = (
                adjusted_weights[asset]
                * total_weight
                / remaining_weight
            )

        self.weights = pd.Series(
            adjusted_weights
        )

    def calculate_metrics(self):
        portfolio_returns = (
            self.weights * self.returns
        ).sum(axis=1)

        annual_returns = (
            portfolio_returns
            .resample('YE')
            .apply(
                lambda x: (1 + x).prod() - 1
            )
        )  # Use 'YE'

        annual_volatility = (
            portfolio_returns
            .resample('YE')
            .std()
            * np.sqrt(252)
        )

        # Avoid division by zero
        sharpe_ratio = (
            annual_returns
            / annual_volatility.replace(0, np.nan)
        )  # Replace zero with NaN to avoid division by zero

        max_drawdown = (
            portfolio_returns.cumsum()
            - portfolio_returns.cumsum().cummax()
        ).min()

        self.metrics = pd.DataFrame({
            'Annual Returns': annual_returns,
            'Annual Volatility': annual_volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown
        })

        self.metrics.to_csv(
            'annual_metrics.csv'
        )

    def plot_net_value(self):
        portfolio_returns = (
            self.weights * self.returns
        ).sum(axis=1)

        net_value = (
            1 + portfolio_returns
        ).cumprod()

        plt.figure(figsize=(10, 6))

        net_value.plot(
            title='Portfolio Net Value (Starting at 1)'
        )

        plt.xlabel('Year')
        plt.ylabel('Net Value')

        plt.axhline(
            1,
            color='red',
            linestyle='--'
        )  # Initial value is 1

        plt.grid()

        plt.savefig(
            'portfolio_net_value.png'
        )

        plt.show()

    def rebalance(self):
        trading_days = self.returns.index.date

        for day in self.rebalance_days:

            if day in trading_days:
                self.allocate_weights(
                    self.select_top_assets()
                )

    def run(self):
        self.generate_virtual_data()
        self.calculate_returns()
        self.calculate_signals()  # Calculate signals
        self.rebalance()  # Rebalance the portfolio

        top_assets = self.select_top_assets()

        self.allocate_weights(
            top_assets
        )

        self.calculate_metrics()
        self.plot_net_value()


# Example usage
if __name__ == "__main__":
    strategy = PortfolioStrategy()
    strategy.run()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class PortfolioStrategy:
    def __init__(self, num_assets=16, start_date='2011-12-31', end_date='2023-12-31',
                 risk_budget=0.1, top_n=11, vol_threshold=0.045, rebalance_days=[4, 5, 6, 7]):
        self.num_assets = num_assets
        self.virtual_start_date = '2009-12-31'
        self.start_date = start_date
        self.end_date = end_date
        self.risk_budget = risk_budget
        self.top_n = top_n  # Set to 11
        self.vol_threshold = vol_threshold
        self.rebalance_days = rebalance_days

        # Define asset names and their maximum position limits
        self.assets_limits = {
            '沪深300': 0.10,
            '中证500': 0.10,
            '中证1000': 0.10,
            '纳斯达克100': 0.10,
            '标普500': 0.10,
            '中国2年国债': 0.40,
            '中国5年国债': 0.40,
            '中国10年国债': 0.40,
            '美国2年国债': 0.60,
            '美国5年国债': 0.40,
            '美国10年国债': 0.40,
            'Brent原油': 0.10,
            '沪金': 0.10,
            '沪铜': 0.10,
            '豆粕': 0.10,
            '黑色系': 0.10
        }
        
        self.price_data = None
        self.returns = None
        self.signals = None
        self.weights = None
        self.metrics = None

    def generate_virtual_data(self):
        workdays = pd.bdate_range(start=self.virtual_start_date, end='2023-12-31')
        num_days = len(workdays)
        dates = pd.date_range(start=self.virtual_start_date, periods=num_days, freq='B')
        assets = list(self.assets_limits.keys())
        
        np.random.seed(42)
        self.price_data = pd.DataFrame(index=dates)

        for asset in assets:
            self.price_data[asset] = 100 * (1 + np.random.normal(0, 0.01, num_days)).cumprod()

        self.price_data.to_csv('virtual_asset_prices-3.0.csv')

    def calculate_returns(self):
        self.returns = self.price_data.pct_change().dropna()
        self.returns = self.returns[self.returns.index >= self.virtual_start_date]

    def calculate_signals(self):
        signals = pd.DataFrame(index=self.returns.index)

        one_month_window = 21
        twelve_month_window = 252  # Approximately 252 trading days in 12 months

        for asset in self.returns.columns:
            # Calculate the 1-month absolute return and 1-month volatility
            monthly_return = self.returns[asset].rolling(window=one_month_window).sum()
            monthly_volatility = self.returns[asset].rolling(window=one_month_window).std()
            signals[f'{asset}_1M'] = monthly_return / monthly_volatility
            
            # Calculate the 12-month absolute return and volatility
            annual_return = self.returns[asset].rolling(window=twelve_month_window).sum()
            annual_volatility = self.returns[asset].rolling(window=twelve_month_window).std()
            signals[f'{asset}_12M/Vol'] = annual_return / annual_volatility
            
            # Calculate the 12-month absolute return
            signals[f'{asset}_12M'] = annual_return
            
            # Calculate (current price - 1-year low) / (1-year high - 1-year low)
            min_price_1y = self.price_data[asset].rolling(window=twelve_month_window).min()
            max_price_1y = self.price_data[asset].rolling(window=twelve_month_window).max()
            signals[f'{asset}_Price_Signal'] = (self.price_data[asset] - min_price_1y) / (max_price_1y - min_price_1y)

            # Calculate the MASS260 signal
            moving_averages = [self.price_data[asset].rolling(window=i).mean() for i in range(1, 261)]
            mass260_scores = [(moving_averages[i] >= moving_averages[i + 1]).astype(int) for i in range(len(moving_averages) - 1)]
            signals[f'{asset}_MASS260'] = pd.concat(mass260_scores, axis=1).mean(axis=1)

        self.signals = signals

    def select_top_assets(self):
        average_signals = self.signals.mean(axis=0)
        asset_averages = {}
        
        for asset in self.returns.columns:
            asset_signals = [
                average_signals[f'{asset}_1M'], 
                average_signals[f'{asset}_12M/Vol'], 
                average_signals[f'{asset}_12M'], 
                average_signals[f'{asset}_Price_Signal'],
            ]
            asset_averages[asset] = np.mean(asset_signals)

        top_assets = sorted(asset_averages, key=asset_averages.get, reverse=True)[:self.top_n]
        return top_assets

    def check_reversal_conditions(self, asset):
        current_mass260 = self.signals[f'{asset}_MASS260'].iloc[-1]
        previous_mass260 = self.signals[f'{asset}_MASS260'].iloc[-2]
        is_low = current_mass260 <= 0.35
        has_reversed = current_mass260 >= previous_mass260 + 0.005

        # Check whether a low point has occurred in the past
        past_mass260 = self.signals[f'{asset}_MASS260'][
            (self.signals[f'{asset}_MASS260'] > 0.35) & 
            (self.signals.index < self.signals.index[-1])
        ]
        has_had_low = (past_mass260.max() <= 0.1)

        return is_low and has_reversed and has_had_low

    def allocate_weights(self, top_assets):
        adjusted_weights = {}
        risk_budget_per_asset = self.risk_budget / self.top_n

        # Initialize the weights of all assets to zero
        for asset in self.assets_limits.keys():
            adjusted_weights[asset] = 0.0

        for asset in top_assets:
            # Calculate the maximum of the historical 22-, 65-, and 130-day volatility
            vol_22 = self.returns[asset].rolling(window=22).std().max()
            vol_65 = self.returns[asset].rolling(window=65).std().max()
            vol_130 = self.returns[asset].rolling(window=130).std().max()
            historical_volatility = max(vol_22, vol_65, vol_130)

            # Reset based on daily volatility observations
            if historical_volatility > 0.045:
                # Reduce the position proportionally to 4%
                adjusted_weights[asset] = 0.04 * (self.assets_limits[asset] / self.assets_limits[asset])
            elif historical_volatility < 0.03:
                adjusted_weights[asset] = 0.04  # Increase the position to 4%
            else:
                # Calculate the correlation between returns and volatility
                correlation = self.returns[asset].corr(
                    self.returns.rolling(window=21).std().mean(axis=1)
                )

                if correlation < 0:
                    # Use more responsive short-term volatility (1 month) to determine the allocation weight
                    short_term_volatility = self.returns[asset].rolling(window=21).std().iloc[-1]
                    adjusted_weights[asset] = (8 * risk_budget_per_asset) / short_term_volatility
                else:
                    # Use a relatively conservative approach based on the maximum of 22/65/130-day volatility
                    adjusted_weights[asset] = (8 * risk_budget_per_asset) / historical_volatility

            # Ensure the weight does not exceed the maximum position limit
            adjusted_weights[asset] = min(
                adjusted_weights[asset], 
                self.assets_limits[asset]
            )

            # Check the reversal mechanism
            if self.check_reversal_conditions(asset):
                # Short-term momentum-based position increase
                short_term_return = self.returns[asset].rolling(window=5).sum().iloc[-1]
                if short_term_return > 0:
                    # Increase the position proportionally by first calculating the current total weight
                    total_weight = sum(adjusted_weights.values())

                    # Increase the weight of the current asset
                    adjusted_weights[asset] += 0.05  # Position increase strategy; can be adjusted as needed

                    # Proportionally reduce the weights of other assets
                    for other_asset in adjusted_weights.keys():
                        if other_asset != asset:
                            adjusted_weights[other_asset] *= (
                                1 - 0.05 / (total_weight - adjusted_weights[asset])
                            )

        # Renormalize the weights so that they sum to 1
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            self.weights = pd.Series({
                asset: weight / total_weight 
                for asset, weight in adjusted_weights.items()
            })
        else:
            self.weights = pd.Series({
                asset: 0.0 
                for asset in adjusted_weights.keys()
            })

    def calculate_metrics(self):
        portfolio_returns = (self.weights * self.returns).sum(axis=1)

        annual_returns = portfolio_returns.resample('YE').apply(
            lambda x: (1 + x).prod() - 1
        )  # Use 'YE'
        annual_volatility = portfolio_returns.resample('YE').std() * np.sqrt(252)

        # Avoid division by zero
        sharpe_ratio = annual_returns / annual_volatility.replace(
            0, np.nan
        )  # Replace 0 with NaN to avoid division by zero

        max_drawdown = (
            portfolio_returns.cumsum() - portfolio_returns.cumsum().cummax()
        ).min()

        self.metrics = pd.DataFrame({
            'Annual Returns': annual_returns,
            'Annual Volatility': annual_volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown
        })
        self.metrics.to_csv('annual_metrics-3.0.csv')

    def plot_net_value(self):
        portfolio_returns = (self.weights * self.returns).sum(axis=1)
        net_value = (1 + portfolio_returns).cumprod()

        plt.figure(figsize=(10, 6))
        net_value.plot(title='Portfolio Net Value (Starting at 1)')
        plt.xlabel('Year')
        plt.ylabel('Net Value')
        plt.axhline(1, color='red', linestyle='--')
        plt.grid()
        plt.savefig('portfolio_net_value-3.0.png')
        plt.show()

    def rebalance(self):
        trading_days = self.returns.index.date
        for day in self.rebalance_days:
            if day in trading_days:
                self.allocate_weights(self.select_top_assets())

    def run(self):
        self.generate_virtual_data()
        self.calculate_returns()
        self.calculate_signals()
        self.rebalance()
        top_assets = self.select_top_assets()
        self.allocate_weights(top_assets)
        self.calculate_metrics()
        self.plot_net_value()


# Example usage

if __name__ == "__main__":
    strategy = PortfolioStrategy()
    strategy.run()

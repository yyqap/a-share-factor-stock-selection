import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# Load all CSV files
def load_data(folder):
    dataframes = []
    for file in os.listdir(folder):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(folder, file))
            df['month'] = file.split('.')[0]  # Extract month information
            dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)


# Create labels
def create_labels(data):
    data['label'] = 0  # Set the default label to 0 (neutral)
    data.loc[data['return'].rank(pct=True) > 0.7, 'label'] = 1   # Positive class
    data.loc[data['return'].rank(pct=True) < 0.3, 'label'] = -1  # Negative class
    return data


# Median-based outlier treatment
def depolarize(data, n=1.5):
    for col in data.columns[4:74]:
        x_M = data[col].median()  # Calculate the median
        D_MAD = (data[col] - x_M).abs().median()  # Calculate the median absolute deviation

        # Apply the outlier treatment
        data[col] = np.where(
            data[col] > x_M + n * D_MAD,
            x_M + n * D_MAD,
            np.where(
                data[col] < x_M - n * D_MAD,
                x_M - n * D_MAD,
                data[col]
            )
        )
    return data


# Standardize features
def standardize_features(data):
    scaler = StandardScaler()
    data.iloc[:, 4:74] = scaler.fit_transform(data.iloc[:, 4:74])  # Standardize features
    return data


# Train and predict
def train_and_predict(train_data, test_data):
    X_train = train_data.iloc[:, 4:74]  # Factor data
    y_train = train_data['label']

    model = LogisticRegression(max_iter=1000)
    model.fit(
        X_train[y_train != 0],
        y_train[y_train != 0]
    )  # Train only on positive and negative classes

    X_test = test_data.iloc[:, 4:74]
    probabilities = model.predict_proba(X_test)[:, 1]  # Get the probability of the positive class

    return probabilities


def calculate_performance_metrics(data, A):
    performance_metrics = []
    cum_return = 1  # Initialize cumulative return
    cum_return_monthly = []
    return_monthly = []

    for month, group in data.groupby('Month'):
        top_a_stocks = group.nlargest(A, 'Probability')

        total_probability = top_a_stocks['Probability'].sum()
        if total_probability == 0:
            continue  # Avoid division by zero

        top_a_stocks['Weight'] = top_a_stocks['Probability'] / total_probability
        top_a_stocks['Weighted Return'] = (
            top_a_stocks['Weight'] * top_a_stocks['return']
        )

        total_return = top_a_stocks['Weighted Return'].sum()
        return_monthly.append(total_return)

        annualized_return = (1 + total_return) ** 12 - 1

        if len(return_monthly) == 1:
            annualized_volatility = 0
        else:
            annualized_volatility = np.std(
                return_monthly,
                ddof=1
            ) * 12 ** 0.5

        sharpe_ratio = (
            annualized_return / annualized_volatility
            if annualized_volatility != 0
            else np.nan
        )

        # Update cumulative return and peak
        cum_return = cum_return * (1 + total_return)
        cum_return_monthly.append(cum_return)

        # Calculate drawdown
        if len(cum_return_monthly) == 1:
            monthly_drawdown = 0
        else:
            monthly_drawdown = (
                (cum_return_monthly[-1] - cum_return_monthly[-2])
                / cum_return_monthly[-2]
            )

        max_drawdown = (
            max(cum_return_monthly) - min(cum_return_monthly)
        ) / max(cum_return_monthly)

        win_rate = (top_a_stocks['return'] > 0).mean()

        performance_metrics.append({
            'Month': month,
            'Annualized Return': annualized_return,
            'Annualized Volatility': annualized_volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Monthly Drawdown': monthly_drawdown,
            'Max Drawdown To Date': max_drawdown,
            'Win Rate': win_rate
        })

    return pd.DataFrame(performance_metrics)


def plot_monthly_portfolio_value(monthly_returns):
    # Calculate monthly portfolio value
    portfolio_values = (1 + monthly_returns).cumprod()

    plt.figure(figsize=(10, 6))
    plt.plot(
        portfolio_values,
        label='Monthly Portfolio Value',
        color='blue'
    )
    plt.title('Monthly Portfolio Value Over Time')
    plt.xlabel('Months')
    plt.ylabel('Portfolio Value')
    plt.axhline(
        1,
        color='red',
        linestyle='--',
        label='Initial Value = 1'
    )
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()
    plt.show()


def calculate_monthly_returns(results_melted, A):
    monthly_returns = []

    for month, group in results_melted.groupby('Month'):
        top_a_stocks = group.nlargest(A, 'Probability')

        total_probability = top_a_stocks['Probability'].sum()

        if total_probability > 0:
            top_a_stocks['Weight'] = (
                top_a_stocks['Probability'] / total_probability
            )

            monthly_return = (
                top_a_stocks['Weight'] * top_a_stocks['return']
            ).sum()

            monthly_returns.append(monthly_return)

        else:
            monthly_returns.append(0)  # Return 0 if there are no available stocks

    return pd.Series(
        monthly_returns,
        index=results_melted['Month'].unique()
    )


def main():
    try:
        folder = '.'  # Set the current directory
        data = load_data(folder)

        # Split the data into training and testing sets
        train_data = data[
            (data['month'].astype(int) >= 82) &
            (data['month'].astype(int) <= 153)
        ]

        test_data = data[
            (data['month'].astype(int) >= 154) &
            (data['month'].astype(int) <= 243)
        ]

        # Filter samples included in the training and testing process
        train_data = train_data[train_data['status'] == 1]
        test_data = test_data[test_data['status'] == 1]

        # Create labels
        train_data = create_labels(train_data)

        # Apply median-based outlier treatment
        train_data = depolarize(train_data)
        test_data = depolarize(test_data)

        # Standardize features
        train_data = standardize_features(train_data)
        test_data = standardize_features(test_data)

        # Get and sort months
        months = sorted(test_data['month'].unique())

        # User inputs the number of stocks A to select each month
        A = int(input("Enter the number of stocks to select each month A: "))

        # Store prediction results for each month
        results = pd.DataFrame(
            index=test_data['stock'].unique(),
            columns=months
        )

        # Define year mapping
        year_mapping = {
            154: 2011, 155: 2011, 156: 2011, 157: 2011, 158: 2011, 159: 2011,
            160: 2011, 161: 2011, 162: 2011, 163: 2011, 164: 2011, 165: 2011,
            166: 2012, 167: 2012, 168: 2012, 169: 2012, 170: 2012, 171: 2012,
            172: 2012, 173: 2012, 174: 2012, 175: 2012, 176: 2012, 177: 2012,
            178: 2013, 179: 2013, 180: 2013, 181: 2013, 182: 2013, 183: 2013,
            184: 2013, 185: 2013, 186: 2013, 187: 2013, 188: 2013, 189: 2013,
            190: 2014, 191: 2014, 192: 2014, 193: 2014, 194: 2014, 195: 2014,
            196: 2014, 197: 2014, 198: 2014, 199: 2014, 200: 2014, 201: 2014,
            202: 2015, 203: 2015, 204: 2015, 205: 2015, 206: 2015, 207: 2015,
            208: 2015, 209: 2015, 210: 2015, 211: 2015, 212: 2015, 213: 2015,
            214: 2016, 215: 2016, 216: 2016, 217: 2016, 218: 2016, 219: 2016,
            220: 2016, 221: 2016, 222: 2016, 223: 2016, 224: 2016, 225: 2016,
            226: 2017, 227: 2017, 228: 2017, 229: 2017, 230: 2017, 231: 2017,
            232: 2017, 233: 2017, 234: 2017, 235: 2017, 236: 2017, 237: 2017,
            238: 2018, 239: 2018, 240: 2018, 241: 2018, 242: 2018, 243: 2018
        }

        for month in months:
            month_data = test_data[test_data['month'] == month]

            if not month_data.empty:
                probabilities = train_and_predict(
                    train_data,
                    month_data
                )

                results.loc[month_data['stock'], month] = probabilities

        # Generate year-month mapping
        results.index.name = 'Stock'

        results.columns = [
            f"{year_mapping[int(month)]}-{(int(month) - 154) % 12 + 1:02d}"
            for month in results.columns
            if str(month).isdigit()
        ]

        results_melted = (
            results
            .reset_index()
            .melt(
                id_vars='Stock',
                var_name='Month',
                value_name='Probability'
            )
        )

        results_melted = results_melted.merge(
            data[['stock', 'return']],
            left_on='Stock',
            right_on='stock',
            how='left'
        )

        # Ensure the Probability column is numeric
        results_melted['Probability'] = pd.to_numeric(
            results_melted['Probability'],
            errors='coerce'
        )

        performance_metrics = calculate_performance_metrics(
            results_melted,
            A
        )

        # Calculate monthly portfolio returns
        monthly_returns = calculate_monthly_returns(
            results_melted,
            A
        )

        # Plot monthly portfolio value
        plot_monthly_portfolio_value(monthly_returns)

        # Save performance metrics
        performance_metrics.to_csv('performance_metrics.csv')

        # Save results with year-month labels
        results.to_csv('predictions_with_years.csv')

    except Exception as e:
        print(f"An error occurred: {e}")


main()

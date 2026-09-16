import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 读取所有 CSV 文件
def load_data(folder):
    dataframes = []
    for file in os.listdir(folder):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(folder, file))
            df['month'] = file.split('.')[0]  # 提取月份信息
            dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)

# 创建标签
def create_labels(data):
    data['label'] = 0  # 默认标签为0（中性）
    data.loc[data['return'].rank(pct=True) > 0.7, 'label'] = 1   # 正例
    data.loc[data['return'].rank(pct=True) < 0.3, 'label'] = -1  # 反例
    return data

# 中位数去极值法
def depolarize(data, n=1.5):
    for col in data.columns[4:74]:
        x_M = data[col].median()  # 计算中位数
        D_MAD = (data[col] - x_M).abs().median()  # 计算中位数绝对偏差

        # 应用去极值法
        data[col] = np.where(data[col] > x_M + n * D_MAD, x_M + n * D_MAD,
                             np.where(data[col] < x_M - n * D_MAD, x_M - n * D_MAD, data[col]))
    return data

# 标准化因子
def standardize_features(data):
    scaler = StandardScaler()
    data.iloc[:, 4:74] = scaler.fit_transform(data.iloc[:, 4:74])  # 标准化特征
    return data

# 训练和预测
def train_and_predict(train_data, test_data):
    X_train = train_data.iloc[:, 4:74]  # 因子数据
    y_train = train_data['label']

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train[y_train != 0], y_train[y_train != 0])  # 只训练正反例

    X_test = test_data.iloc[:, 4:74]
    probabilities = model.predict_proba(X_test)[:, 1]  # 获取正例的概率
    
    return probabilities

def calculate_performance_metrics(data, A):
    performance_metrics = []
    cum_return = 1  # 初始化累计收益
    cum_return_monthly=[]
    return_monthly=[]
    
    for month, group in data.groupby('Month'):
        top_a_stocks = group.nlargest(A, 'Probability')
        
        total_probability = top_a_stocks['Probability'].sum()
        if total_probability == 0:
            continue  # 避免除以零
        
        top_a_stocks['Weight'] = top_a_stocks['Probability'] / total_probability
        top_a_stocks['Weighted Return'] = top_a_stocks['Weight'] * top_a_stocks['return']
        
        total_return = top_a_stocks['Weighted Return'].sum()
        return_monthly.append(total_return)
        annualized_return = (1+total_return) ** 12-1
        if len(return_monthly)==1:
            annualized_volatility=0
        else:    
            annualized_volatility = np.std(return_monthly, ddof=1)*12**0.5
        
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else np.nan
        
        # 更新累计收益和峰值
        cum_return = cum_return * (1 + total_return)
        cum_return_monthly.append(cum_return)

        # 计算回撤
        if len(cum_return_monthly)==1:
            monthly_drawdown=0
        else:
            monthly_drawdown = (cum_return_monthly[-1]-cum_return_monthly[-2])/cum_return_monthly[-2]
        max_drawdown = (max(cum_return_monthly)- min(cum_return_monthly))/max(cum_return_monthly)
        
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
    # 计算每月组合净值
    portfolio_values = (1 + monthly_returns).cumprod()  # 计算组合净值
    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_values, label='Monthly Portfolio Value', color='blue')
    plt.title('Monthly Portfolio Value Over Time')
    plt.xlabel('Months')
    plt.ylabel('Portfolio Value')
    plt.axhline(1, color='red', linestyle='--', label='Initial Value = 1')
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
            top_a_stocks['Weight'] = top_a_stocks['Probability'] / total_probability
            monthly_return = (top_a_stocks['Weight'] * top_a_stocks['return']).sum()
            monthly_returns.append(monthly_return)
        else:
            monthly_returns.append(0)  # 如果没有可用的股票，返回0

    return pd.Series(monthly_returns, index=results_melted['Month'].unique())

def main():
    try:
        folder = '.'  # 设置为当前目录
        data = load_data(folder)

        # 将数据分为训练集和验证集
        train_data = data[(data['month'].astype(int) >= 82) & (data['month'].astype(int) <= 153)]
        test_data = data[(data['month'].astype(int) >= 154) & (data['month'].astype(int) <= 243)]

        # 过滤参与训练的样本
        train_data = train_data[train_data['status'] == 1]
        test_data = test_data[test_data['status'] == 1]

        # 创建标签
        train_data = create_labels(train_data)

        # 应用中位数去极值法
        train_data = depolarize(train_data)
        test_data = depolarize(test_data)

        # 标准化因子
        train_data = standardize_features(train_data)
        test_data = standardize_features(test_data)

        # 获取月份并排序
        months = sorted(test_data['month'].unique())

        # 用户输入每月选股的数量 A
        A = int(input("请输入每月选股的数量 A: "))

        # 存储每个月的预测结果
        results = pd.DataFrame(index=test_data['stock'].unique(), columns=months)

        # 定义年份映射
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
                probabilities = train_and_predict(train_data, month_data)
                results.loc[month_data['stock'], month] = probabilities


        # 生成年份和月份的映射
        results.index.name = 'Stock'
        results.columns = [
            f"{year_mapping[int(month)]}-{(int(month) - 154) % 12 + 1:02d}" 
            for month in results.columns if month.isdigit()
        ]


 
        results_melted = results.reset_index().melt(id_vars='Stock', var_name='Month', value_name='Probability')
        results_melted = results_melted.merge(data[['stock', 'return']], left_on='Stock', right_on='stock', how='left')
        # 确保 Probability 列是数值型
        results_melted['Probability'] = pd.to_numeric(results_melted['Probability'], errors='coerce')
        
        performance_metrics = calculate_performance_metrics(results_melted,A)

        # 计算每月组合收益率
        monthly_returns = calculate_monthly_returns(results_melted, A)

        # 绘制每月组合净值
        plot_monthly_portfolio_value(monthly_returns)
        # 保存绩效指标
        performance_metrics.to_csv('performance_metrics.csv')

        # 保存结果为包含年份的 CSV 文件
        results.to_csv('predictions_with_years.csv')

    except Exception as e:
        print(f"An error occurred: {e}")

main()

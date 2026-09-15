"""A-share factor stock selection with logistic regression.

The script expects monthly cross-sectional CSV files with columns:
month, stock, status, return, followed by factor columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def load_panel(data_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for csv_path in sorted(data_dir.glob("*.csv"), key=lambda p: int(p.stem)):
        frame = pd.read_csv(csv_path)
        if "month" not in frame.columns:
            frame["month"] = int(csv_path.stem)
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    panel = pd.concat(frames, ignore_index=True)
    panel["month"] = panel["month"].astype(int)
    return panel


def factor_columns(panel: pd.DataFrame) -> list[str]:
    required = {"month", "stock", "status", "return"}
    return [col for col in panel.columns if col not in required]


def label_by_month(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    panel["label"] = 0

    def assign(group: pd.DataFrame) -> pd.DataFrame:
        pct_rank = group["return"].rank(pct=True)
        group.loc[pct_rank > 0.7, "label"] = 1
        group.loc[pct_rank < 0.3, "label"] = -1
        return group

    return panel.groupby("month", group_keys=False).apply(assign)


def winsorize_mad(panel: pd.DataFrame, factors: list[str], threshold: float = 1.5) -> pd.DataFrame:
    panel = panel.copy()

    def winsorize(group: pd.DataFrame) -> pd.DataFrame:
        medians = group[factors].median()
        mad = (group[factors] - medians).abs().median().replace(0, np.nan)
        lower = medians - threshold * mad
        upper = medians + threshold * mad
        group[factors] = group[factors].clip(lower=lower, upper=upper, axis=1)
        return group

    return panel.groupby("month", group_keys=False).apply(winsorize)


def prepare_train_test(
    panel: pd.DataFrame,
    train_start: int,
    train_end: int,
    test_start: int,
    test_end: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    factors = factor_columns(panel)
    panel = panel[panel["status"] == 1].copy()
    panel = panel.dropna(subset=factors + ["return"])
    panel = winsorize_mad(panel, factors)

    train = panel[(panel["month"] >= train_start) & (panel["month"] <= train_end)].copy()
    test = panel[(panel["month"] >= test_start) & (panel["month"] <= test_end)].copy()
    train = label_by_month(train)

    scaler = StandardScaler()
    train[factors] = scaler.fit_transform(train[factors])
    test[factors] = scaler.transform(test[factors])

    return train, test, factors


def fit_model(train: pd.DataFrame, factors: list[str]) -> LogisticRegression:
    labeled = train[train["label"] != 0]
    model = LogisticRegression(max_iter=1000)
    model.fit(labeled[factors], labeled["label"])
    return model


def predict_probabilities(test: pd.DataFrame, factors: list[str], model: LogisticRegression) -> pd.DataFrame:
    scored = test[["month", "stock", "return"]].copy()
    scored["probability"] = model.predict_proba(test[factors])[:, 1]
    return scored


def probability_matrix(scored: pd.DataFrame) -> pd.DataFrame:
    matrix = scored.pivot(index="stock", columns="month", values="probability")
    matrix = matrix.sort_index(axis=0).sort_index(axis=1)
    return matrix


def backtest_top_n(scored: pd.DataFrame, top_n: int) -> tuple[pd.DataFrame, pd.Series]:
    monthly_returns: list[tuple[int, float]] = []

    for month, group in scored.groupby("month"):
        selected = group.nlargest(top_n, "probability").copy()
        total_probability = selected["probability"].sum()
        if total_probability <= 0:
            monthly_returns.append((month, 0.0))
            continue

        weights = selected["probability"] / total_probability
        monthly_return = float((weights * selected["return"]).sum())
        monthly_returns.append((month, monthly_return))

    returns = pd.Series(
        data=[value for _, value in monthly_returns],
        index=[month for month, _ in monthly_returns],
        name="monthly_return",
    )
    net_value = (1 + returns).cumprod()
    running_max = net_value.cummax()
    drawdown = net_value / running_max - 1

    metrics = pd.DataFrame(
        {
            "month": returns.index,
            "monthly_return": returns.values,
            "net_value": net_value.values,
            "annualized_return_to_date": net_value.values ** (12 / np.arange(1, len(net_value) + 1)) - 1,
            "annualized_volatility_to_date": returns.expanding(2).std().values * np.sqrt(12),
            "max_drawdown_to_date": drawdown.expanding().min().values,
        }
    )
    metrics["sharpe_to_date"] = (
        metrics["annualized_return_to_date"] / metrics["annualized_volatility_to_date"]
    )

    return metrics, net_value


def plot_net_value(net_value: pd.Series, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    net_value.plot(ax=ax, color="#2563eb", linewidth=2)
    ax.axhline(1.0, color="#dc2626", linestyle="--", linewidth=1)
    ax.set_title("Monthly Portfolio Net Value")
    ax.set_xlabel("Month index")
    ax.set_ylabel("Net value")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and backtest an A-share factor stock-selection model.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing monthly factor CSV files.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/task1"), help="Directory for outputs.")
    parser.add_argument("--top-n", type=int, default=30, help="Number of stocks selected each month.")
    parser.add_argument("--train-start", type=int, default=82, help="First in-sample month index.")
    parser.add_argument("--train-end", type=int, default=153, help="Last in-sample month index.")
    parser.add_argument("--test-start", type=int, default=154, help="First out-of-sample month index.")
    parser.add_argument("--test-end", type=int, default=243, help="Last out-of-sample month index.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    panel = load_panel(args.data_dir)
    train, test, factors = prepare_train_test(
        panel,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
    )
    model = fit_model(train, factors)
    scored = predict_probabilities(test, factors, model)

    probability_matrix(scored).to_csv(args.output_dir / "predictions_with_years.csv")
    metrics, net_value = backtest_top_n(scored, args.top_n)
    metrics.to_csv(args.output_dir / f"performance_metrics_top_{args.top_n}.csv", index=False)
    plot_net_value(net_value, args.output_dir / f"monthly_portfolio_value_top_{args.top_n}.png")


if __name__ == "__main__":
    main()

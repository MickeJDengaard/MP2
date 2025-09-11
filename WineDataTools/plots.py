import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


# -------------------------------
# Histograms
# -------------------------------
def show_histograms(df: pd.DataFrame, bins: int = 10, layout: str = "separate", bell_curve: bool = False):
    numeric_cols = df.select_dtypes(include="number").columns

    if layout == "grid":
        n = len(numeric_cols)
        rows = (n + 2) // 3
        fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))
        axes = axes.flatten()
        for i, col in enumerate(numeric_cols):
            data = df[col].dropna()
            axes[i].hist(data, bins=bins, density=True, alpha=0.7, color='tab:blue', edgecolor='black')
            if bell_curve and len(data) > 1:
                mu, std = data.mean(), data.std()
                x = np.linspace(data.min(), data.max(), 100)
                pdf = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / std) ** 2)
                axes[i].plot(x, pdf, 'r', linewidth=2)
            axes[i].set_title(f'Histogram of {col}')
        plt.tight_layout()
        plt.show()
    else:
        for col in numeric_cols:
            data = df[col].dropna()
            plt.figure(figsize=(6, 4))
            plt.hist(data, bins=bins, density=True, alpha=0.7, color='tab:blue', edgecolor='black')
            if bell_curve and len(data) > 1:
                mu, std = data.mean(), data.std()
                x = np.linspace(data.min(), data.max(), 100)
                pdf = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / std) ** 2)
                plt.plot(x, pdf, 'r', linewidth=2)
            plt.title(f'Histogram of {col}')
            plt.show()


def show_grouped_histograms(df: pd.DataFrame, category_col="type", bins=10, layout="grid", bell_curve=True, max_cols=3):
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c != category_col]
    if category_col not in df.columns:
        raise ValueError(f"Category column '{category_col}' not found in dataframe")

    cats = df[category_col].dropna().unique()
    palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_map = {cat: palette[i % len(palette)] for i, cat in enumerate(cats)}

    def _plot_column(ax, col):
        bin_edges = np.histogram_bin_edges(df[col].dropna(), bins=bins)
        width = np.diff(bin_edges)[0] / len(cats)
        for i, cat in enumerate(cats):
            vals = df[df[category_col] == cat][col].dropna()
            counts, _ = np.histogram(vals, bins=bin_edges, density=True)
            ax.bar(bin_edges[:-1] + i * width, counts, width=width, color=color_map[cat], alpha=0.7,
                   edgecolor='black', label=str(cat))
            if bell_curve and len(vals) > 1:
                mu, std = vals.mean(), vals.std()
                x = np.linspace(bin_edges[0], bin_edges[-1], 100)
                pdf = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / std) ** 2)
                ax.plot(x, pdf, color=color_map[cat])
        ax.set_title(f"Histogram of {col} by {category_col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        ax.legend()

    if layout == "grid":
        n = len(numeric_cols)
        cols = min(max_cols, n)
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows), squeeze=False)
        axes = axes.flatten()
        for i, col in enumerate(numeric_cols):
            _plot_column(axes[i], col)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        plt.show()
    else:
        for col in numeric_cols:
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            _plot_column(ax, col)
            plt.tight_layout()
            plt.show()


# -------------------------------
# Boxplots
# -------------------------------
def show_boxplots(df: pd.DataFrame, layout: str = "grid", category_col=None):
    numeric_cols = df.select_dtypes(include="number").columns
    if category_col and category_col in numeric_cols:
        numeric_cols = numeric_cols.drop(category_col)

    if layout == "grid":
        n = len(numeric_cols)
        rows = (n + 2) // 3
        fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))
        axes = axes.flatten()
        for i, col in enumerate(numeric_cols):
            df.boxplot(column=col, ax=axes[i])
        plt.tight_layout()
        plt.show()
    else:
        for col in numeric_cols:
            df.boxplot(column=col)
            plt.show()


def boxplots_by_type(df: pd.DataFrame, category_col="type"):
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c != category_col]
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=df, x=category_col, y=col)
        plt.title(f"Boxplot of {col} by {category_col}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# -------------------------------
# Scatterplots
# -------------------------------
def scatter_vs_quality(df: pd.DataFrame, features=None, target="quality", category_col="type"):
    if features is None:
        features = ["alcohol", "residual sugar", "pH"]

    for feature in features:
        if feature in df.columns and target in df.columns:
            plt.figure(figsize=(6, 4))
            sns.scatterplot(data=df, x=feature, y=target, hue=category_col, alpha=0.7)
            plt.title(f"{feature} vs {target}")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()


# -------------------------------
# Correlation heatmap
# -------------------------------
def show_correlation_heatmap(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include="number").columns
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()


# -------------------------------
# Mean comparisons
# -------------------------------
def mean_comparison(df: pd.DataFrame, features=None, category_col="type"):
    if features is None:
        features = df.select_dtypes(include="number").columns.tolist()
        if category_col in features:
            features.remove(category_col)

    means = df.groupby(category_col)[features].mean()
    if set(["white", "red"]).issubset(means.index):
        diffs = means.loc["white"] - means.loc["red"]
    else:
        diffs = pd.Series(dtype=float)

    return means, diffs

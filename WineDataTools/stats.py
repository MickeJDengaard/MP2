import pandas as pd

def describe_wine_data(df: pd.DataFrame, verbose: bool = True, round_digits: int = 0) -> pd.DataFrame:
    """Genererer deskriptiv statistik for datasættet."""
    wine_summary = df.describe(include='all').transpose()
    wine_summary = wine_summary.drop(columns=[c for c in ['unique','top','freq'] if c in wine_summary.columns], errors="ignore")

    if round_digits > 0:
        wine_summary = wine_summary.round(round_digits)

    if 'count' in wine_summary.columns:
        wine_summary['count'] = wine_summary['count'].astype('Int64')

    wine_summary.index.name = "feature"

    if verbose:
        print(f"Summary: {df.shape[0]} rows × {df.shape[1]} columns")
        print(wine_summary.to_string())

    return wine_summary


def mean_comparison(df: pd.DataFrame, category_col: str = "type"):
    """
    Beregner gennemsnit for alle numeriske kolonner opdelt på kategori,
    samt forskellen mellem de to kategorier.

    Args:
        df (pd.DataFrame): Input DataFrame med wine data.
        category_col (str): Kolonne med kategori ('type' som standard).

    Returns:
        tuple: (means_df, diff_df)
            means_df: DataFrame med gennemsnit pr. kategori
            diff_df: Series med forskel white - red
    """
    numeric_cols = df.select_dtypes(include="number").columns
    means = df.groupby(category_col)[numeric_cols].mean()
    
    if 'white' in means.index and 'red' in means.index:
        diff = means.loc['white'] - means.loc['red']
    else:
        diff = pd.Series(dtype=float)

    return means, diff
    

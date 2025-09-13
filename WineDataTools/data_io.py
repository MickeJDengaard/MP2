import pandas as pd

def read_wine_data(file_path, winetype) -> pd.DataFrame:

    # Reads our .xlsx file and sets the header to row 1 so that we don't get any empty rows
    df = pd.read_excel(file_path, header=1)
    
    # Removeing all columns that start with "Unnamed" | To get rid of any columns that are not needed in our data analysis
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Converting all columns to numeric values so that we can perform calculations and to now what type of data we have
    df = df.apply(pd.to_numeric, errors='coerce')

    # Adding type-column
    df['type'] = winetype

    # Drop rows, where all values are NaN | To get rid of any rows that are not needed in our data analysis
    df.dropna(how='all', inplace=True)

    return df

def combine_dataframes(dfs) -> pd.DataFrame:
    """Combines multiple dataframes into one with clean index."""
    return pd.concat(dfs, ignore_index=True).reset_index(drop=True)

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Removes duplicates and reindexes."""
    return df.drop_duplicates().reset_index(drop=True)

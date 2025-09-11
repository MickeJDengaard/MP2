import pandas as pd

def read_wine_data(file_path, winetype) -> pd.DataFrame:
    """
    Indlæser Excel-fil, renser data:
      - Hopper over tomme rækker/headers
      - Konverterer numeriske felter
      - Fjerner 'Unnamed:' kolonner
      - Tilføjer type-kolonne
    """
    # Læs Excel, header på række 1 (hopper evt. over tom første række)
    df = pd.read_excel(file_path, header=1)
    
    # Fjern kolonner, der starter med "Unnamed"
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Konverter alle kolonner til numeriske værdier (NaN hvis ikke muligt)
    df = df.apply(pd.to_numeric, errors='coerce')

    # Tilføj type-kolonne
    df['type'] = winetype

    # Drop rækker, hvor alle værdier er NaN
    df.dropna(how='all', inplace=True)

    return df

def combine_dataframes(dfs) -> pd.DataFrame:
    """Kombinerer flere dataframes til én med clean index."""
    return pd.concat(dfs, ignore_index=True).reset_index(drop=True)

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Fjerner duplikater og genindekserer."""
    return df.drop_duplicates().reset_index(drop=True)

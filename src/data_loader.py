import pandas as pd
from pandas import DataFrame

def load_wine_data( path: str = "data/winequality-red.csv") -> DataFrame: 

    df = pd.read_csv(path)
    return df
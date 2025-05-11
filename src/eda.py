import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_wine_data

def run_eda():
    df = load_wine_data()

    print("Informacoes do data frame")
    print(df.info(), "\n")

    print("Estatistica Descritivas")
    print(df.describe(), "\n")

    print("Valores faltantes por coluna")
    print(df.isnull().sum, "\n")

    df.hist(bins=20, figsize=(12, 8))
    plt.tight_layout()
    plt.show()

    corr = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.show()

if __name__ == "__main__":
    run_eda()
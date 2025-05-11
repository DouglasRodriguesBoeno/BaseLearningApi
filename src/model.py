from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from data_loader import load_wine_data

def tune_hyperparameters(path_csv: str = "data/winequality-red.csv"):
    df = load_wine_data(path_csv)
    X = df.drop("quality", axis=1)
    y = df["quality"]

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10]
    }

    grid = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid,
        cv=5,
        scoring="r2",
        n_jobs=1
    )
    grid.fit(X, y)

    print("🔍 Melhores parâmetros encontrados:")

    for K, v in grid.best_params_.items(): 
        print(f"  • {K}: {v}")
    print(f"👍 Melhor R² médio (5-fold): {grid.best_score_:.3f}")

    joblib.dump(grid.best_estimator_, "model_tuned.joblib")
    print("Modelo ajustado salvo em 'model_tuned.joblib'")

def evalue_modal_cv(path_csv: str = "data/winequality-red.csv"):

    df = load_wine_data(path_csv)
    X = df.drop("quality", axis=1)
    y = df["quality"]

    model = RandomForestRegressor(n_estimators=100, random_state=42)

    scores = cross_val_score(model, X, y, cv=5, scoring="r2", n_jobs=1)
    print(f"[CV] R² médio (5-fold): {scores.mean():.3f} ± {scores.std():.3f}")

def train_and_save(path_csv: str = "data/winequality-red.csv",
                    model_path: str = "model.joblib"):
    
    df = load_wine_data(path_csv)
    X = df.drop("quality", axis=1)
    y = df["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model_tuned = joblib.load("model_tuned.joblib")

    preds = model_tuned.predict((X_test))
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"[Tuned Model] MSE: {mse:.3f} — R²: {r2:.3f}")

    joblib.dump(model_tuned, model_path)
    print(f"[Model] Modelo salvo em {model_path}")
    
if __name__ == "__main__":
    evalue_modal_cv()
    tune_hyperparameters()
    train_and_save()





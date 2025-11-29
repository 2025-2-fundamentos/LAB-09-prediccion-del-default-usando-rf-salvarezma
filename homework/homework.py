import gzip
import json
import os
import pickle
import zipfile
from glob import glob
from pathlib import Path

import pandas as pd  # type: ignore
from sklearn.compose import ColumnTransformer  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.metrics import (  # type: ignore
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.preprocessing import OneHotEncoder  # type: ignore

def leer_zip_a_dfs(directorio: str) -> list[pd.DataFrame]:
    dataframes = []
    for zip_path in glob(os.path.join(directorio, "*")):
        with zipfile.ZipFile(zip_path, "r") as zf:
            for miembro in zf.namelist():
                with zf.open(miembro) as fh:
                    dataframes.append(pd.read_csv(fh, sep=",", index_col=0))
    return dataframes

def reiniciar_directorio(ruta: str) -> None:
    if os.path.exists(ruta):
        for f in glob(os.path.join(ruta, "*")):
            try:
                os.remove(f)
            except IsADirectoryError:
                pass
        try:
            os.rmdir(ruta)
        except OSError:
            pass
    os.makedirs(ruta, exist_ok=True)


def guardar_modelo_gz(ruta_salida: str, objeto) -> None:
    reiniciar_directorio(os.path.dirname(ruta_salida))
    with gzip.open(ruta_salida, "wb") as fh:
        pickle.dump(objeto, fh)

def depurar(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp = tmp.rename(columns={"default payment next month": "default"})

    tmp = tmp.loc[tmp["MARRIAGE"] != 0]
    tmp = tmp.loc[tmp["EDUCATION"] != 0]

    tmp["EDUCATION"] = tmp["EDUCATION"].apply(lambda v: 4 if v >= 4 else v)

    return tmp.dropna()

def separar_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=["default"])
    y = df["default"]
    return X, y

def ensamblar_busqueda() -> GridSearchCV:
    cat_cols = ["SEX", "EDUCATION", "MARRIAGE"]
    ohe = OneHotEncoder(handle_unknown="ignore")
    ct = ColumnTransformer(
        transformers=[("cat", ohe, cat_cols)],
        remainder="passthrough",
    )

    clf = RandomForestClassifier(random_state=42)
    pipe = Pipeline(
        steps=[
            ("prep", ct),
            ("rf", clf),
        ]
    )

    grid_params = {
        "rf__n_estimators": [100, 200, 500],
        "rf__max_depth": [None, 5, 10],
        "rf__min_samples_split": [2, 5],
        "rf__min_samples_leaf": [1, 2],
    }

    gs = GridSearchCV(
        estimator=pipe,
        param_grid=grid_params,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
        refit=True,
        verbose=2,
    )
    return gs

def empaquetar_metricas(etiqueta: str, y_true, y_pred) -> dict:
    return {
        "dataset": etiqueta,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }

def empaquetar_matriz_conf(etiqueta: str, y_true, y_pred) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    return {
        "type": "cm_matrix",
        "dataset": etiqueta,
        "true_0": {"predicted_0": int(cm[0][0]), "predicted_1": int(cm[0][1])},
        "true_1": {"predicted_0": int(cm[1][0]), "predicted_1": int(cm[1][1])},
    }

def main() -> None:
    df_list = [depurar(d) for d in leer_zip_a_dfs("files/input")]

    test_df, train_df = df_list

    X_tr, y_tr = separar_xy(train_df)
    X_te, y_te = separar_xy(test_df)

    buscador = ensamblar_busqueda()
    buscador.fit(X_tr, y_tr)

    guardar_modelo_gz(os.path.join("files", "models", "model.pkl.gz"), buscador)

    yhat_test = buscador.predict(X_te)
    yhat_train = buscador.predict(X_tr)

    m_test = empaquetar_metricas("test", y_te, yhat_test)
    m_train = empaquetar_metricas("train", y_tr, yhat_train)

    cm_test = empaquetar_matriz_conf("test", y_te, yhat_test)
    cm_train = empaquetar_matriz_conf("train", y_tr, yhat_train)

    Path("files/output").mkdir(parents=True, exist_ok=True)
    with open("files/output/metrics.json", "w", encoding="utf-8") as fh:
        fh.write(json.dumps(m_train) + "\n")
        fh.write(json.dumps(m_test) + "\n")
        fh.write(json.dumps(cm_train) + "\n")
        fh.write(json.dumps(cm_test) + "\n")

if __name__ == "__main__":
    main()
# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

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
        "type": "metrics",
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
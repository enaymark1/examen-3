"""Aplica el modelo guardado por train.py a test y genera submissions/06_lgbm_svd.csv.

Lee también train porque algunas features sin respuesta se calculan sobre train + test juntos
(conteos y perfil del vendedor, SVD del texto). De train no usa ninguna respuesta: todo lo aprendido
de la respuesta viene de modelo/modelo_final.pkl.
"""
import os
import pickle

import numpy as np
import pandas as pd

from features import (cargar_datos, armar_tabla, componentes_svd, texto_combinado, claves_cruzadas,
                      aplicar_target_encoding, fijar_categorias)

with open("modelo/modelo_final.pkl", "rb") as f:
    modelo = pickle.load(f)

todo, y = cargar_datos()
N_TRAIN = len(y)
test = lambda s: s.iloc[N_TRAIN:].reset_index(drop=True)

X_todo = armar_tabla(todo)
X_svd = pd.concat([X_todo, componentes_svd(todo, 128)], axis=1)

# piezas de test con lo aprendido en train
claves = {"te_seller_id": todo["seller_id"], "te_category_id": todo["category_id"], **claves_cruzadas(todo)}
piezas_test = {n: aplicar_target_encoding(modelo["tablas"][n], modelo["prior"], test(c)) for n, c in claves.items()}
piezas_test["prob_combinado"] = modelo["texto_modelo"].predict_proba(test(texto_combinado(todo)))[:, 1]

# promedio de los dos LightGBM
p = np.zeros(len(todo) - N_TRAIN)
for X, lgbm, columnas in zip([X_todo, X_svd], modelo["lgbm"], modelo["columnas"]):
    B = X.iloc[N_TRAIN:].reset_index(drop=True)
    for nombre, valores in piezas_test.items():
        B[nombre] = valores
    B = fijar_categorias(B[columnas], modelo["categorias"])
    p += lgbm.predict_proba(B)[:, 1] / 2

os.makedirs("submissions", exist_ok=True)
pd.DataFrame({"ID": range(1, len(p) + 1), "condition": np.where(p > 0.5, "used", "new")}).to_csv(
    "submissions/06_lgbm_svd.csv", index=False)
print("submission guardada en submissions/06_lgbm_svd.csv")

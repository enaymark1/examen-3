"""Aplica el modelo guardado por train.py a test y genera submissions/08_ensamble_4_nbsvm_cat_dart.csv.

Lee también train porque algunas features sin respuesta se calculan sobre train + test juntos
(conteos y perfil del vendedor, SVD del texto, vocabulario de NB-SVM). De train no usa ninguna respuesta:
todo lo aprendido de la respuesta viene de modelo/modelo_final.pkl.
"""
import os
import pickle

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from features import (cargar_datos, armar_tabla, componentes_svd, texto_combinado, claves_cruzadas,
                      aplicar_target_encoding, matriz_palabras, aplicar_nbsvm, preparar_catboost, fijar_categorias)

with open("modelo/modelo_final.pkl", "rb") as f:
    modelo = pickle.load(f)

todo, y = cargar_datos()
N_TRAIN = len(y)
test = lambda s: s.iloc[N_TRAIN:].reset_index(drop=True)

X_svd = pd.concat([armar_tabla(todo), componentes_svd(todo, 128)], axis=1)
M = matriz_palabras(todo)

# test con todas las columnas posibles: features + SVD + piezas aprendidas en train
B_todo = test(X_svd)
claves = {"te_seller_id": todo["seller_id"], "te_category_id": todo["category_id"], **claves_cruzadas(todo)}
for nombre, c in claves.items():
    B_todo[nombre] = aplicar_target_encoding(modelo["tablas"][nombre], modelo["prior"], test(c))
B_todo["prob_combinado"] = modelo["texto_modelo"].predict_proba(test(texto_combinado(todo)))[:, 1]
B_todo["prob_nbsvm"] = aplicar_nbsvm(modelo["nbsvm"], M[N_TRAIN:])

# cada modelo toma sus columnas, en el mismo orden que en el entrenamiento; después, promedio simple
probs = []
for m, columnas in zip(modelo["modelos"], modelo["columnas"]):
    B = fijar_categorias(B_todo[columnas].copy(), modelo["categorias"])
    if isinstance(m, CatBoostClassifier):
        B = preparar_catboost(B)
    probs.append(m.predict_proba(B)[:, 1])
p = sum(probs) / len(probs)

os.makedirs("submissions", exist_ok=True)
pd.DataFrame({"ID": range(1, len(p) + 1), "condition": np.where(p > 0.5, "used", "new")}).to_csv(
    "submissions/08_ensamble_4_nbsvm_cat_dart.csv", index=False)
print("submission guardada en submissions/08_ensamble_4_nbsvm_cat_dart.csv")

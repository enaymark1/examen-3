"""Entrena el modelo final (notebook 06) con los 70.000 ítems de train y lo guarda en modelo/modelo_final.pkl.

Modelo: promedio de dos LightGBM.
  - LightGBM A: 92 features sin respuesta + 6 piezas con respuesta (5 target encodings + probabilidad del texto).
  - LightGBM B: lo mismo + las 128 columnas de la SVD del texto.

Lo que se guarda es todo lo que se aprende de la respuesta y hace falta para predecir test:
las tablas de los target encodings, el modelo de texto, las categorías de entrenamiento y los dos LightGBM.
"""
import os
import pickle

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from features import (CAT, cargar_datos, armar_tabla, componentes_svd, texto_combinado, claves_cruzadas,
                      target_encoding, tabla_target_encoding, prob_texto_fuera_de_fold, modelo_texto,
                      fijar_categorias)

PARAMS_LGBM = dict(n_estimators=1500, learning_rate=0.02, num_leaves=63, colsample_bytree=0.5, random_state=42, verbose=-1)

todo, y = cargar_datos()
N_TRAIN = len(y)
train = lambda s: s.iloc[:N_TRAIN].reset_index(drop=True)

# features sin respuesta: se calculan sobre train + test juntos (conteos, perfil del vendedor, SVD)
X_todo = armar_tabla(todo)
X_svd = pd.concat([X_todo, componentes_svd(todo, 128)], axis=1)

# piezas con respuesta, en train fuera de fold; en el mismo orden que en los notebooks
claves = {"te_seller_id": todo["seller_id"], "te_category_id": todo["category_id"], **claves_cruzadas(todo)}
texto = texto_combinado(todo)
piezas_train = {}
for nombre in ["te_seller_id", "te_category_id", "prob_combinado", *claves_cruzadas(todo)]:
    if nombre == "prob_combinado":
        piezas_train[nombre] = prob_texto_fuera_de_fold(train(texto), y, caracteres=True)
    else:
        piezas_train[nombre] = target_encoding(train(claves[nombre]), y, train(claves[nombre]).iloc[:0])[0]

# lo que se aplica a test: tablas de target encoding y modelo de texto, ajustados con todo train
prior = y.mean()
tablas = {n: tabla_target_encoding(train(c), y, prior) for n, c in claves.items()}
texto_modelo = modelo_texto(caracteres=True).fit(train(texto), y)

# los dos LightGBM
lgbm, columnas = [], []
for X in [X_todo, X_svd]:
    A = X.iloc[:N_TRAIN].reset_index(drop=True)
    for nombre, valores in piezas_train.items():
        A[nombre] = valores
    categorias = {c: pd.Categorical(A[c]).categories for c in CAT}
    A = fijar_categorias(A, categorias)
    lgbm.append(LGBMClassifier(**PARAMS_LGBM).fit(A, y))
    columnas.append(list(A.columns))

os.makedirs("modelo", exist_ok=True)
with open("modelo/modelo_final.pkl", "wb") as f:
    pickle.dump({"prior": prior, "tablas": tablas, "texto_modelo": texto_modelo, "categorias": categorias,
                 "lgbm": lgbm, "columnas": columnas}, f)
print("modelo guardado en modelo/modelo_final.pkl")

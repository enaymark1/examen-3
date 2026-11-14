"""Entrena el modelo final (ensamble 08) con los 70.000 ítems de train y lo guarda en modelo/modelo_final.pkl.

Modelo: promedio simple de cuatro modelos de boosting.
  - LightGBM A:    92 features sin respuesta + 6 piezas con respuesta + probabilidad NB-SVM del texto.
  - LightGBM B:    lo mismo + las 128 columnas de la SVD del texto.
  - CatBoost:      92 features + 6 piezas + SVD (sin NB-SVM).
  - LightGBM DART: 92 features + 6 piezas + SVD (sin NB-SVM).
Las 6 piezas son 5 target encodings (vendedor, categoría y 3 cruces) y la probabilidad del texto (TF-IDF + logística).

Lo que se guarda es todo lo que se aprende de la respuesta y hace falta para predecir test:
las tablas de los target encodings, los dos modelos de texto, las categorías de entrenamiento y los cuatro modelos.
"""
import os
import pickle

import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from features import (CAT, cargar_datos, armar_tabla, componentes_svd, texto_combinado, claves_cruzadas,
                      target_encoding, tabla_target_encoding, prob_texto_fuera_de_fold, modelo_texto,
                      matriz_palabras, ajustar_nbsvm, prob_nbsvm_fuera_de_fold, preparar_catboost, fijar_categorias)

PARAMS_LGBM = dict(n_estimators=1500, learning_rate=0.02, num_leaves=63, colsample_bytree=0.5, random_state=42, verbose=-1)
PARAMS_DART = {**PARAMS_LGBM, "boosting_type": "dart", "n_estimators": 800, "learning_rate": 0.08, "drop_rate": 0.1}
PARAMS_CAT = dict(iterations=2000, learning_rate=0.06, depth=7, random_seed=42, verbose=0, thread_count=8,
                  allow_writing_files=False)

todo, y = cargar_datos()
N_TRAIN = len(y)
train = lambda s: s.iloc[:N_TRAIN].reset_index(drop=True)

# features sin respuesta: se calculan sobre train + test juntos (conteos, perfil del vendedor, SVD, vocabulario)
X_todo = armar_tabla(todo)
X_svd = pd.concat([X_todo, componentes_svd(todo, 128)], axis=1)
M = matriz_palabras(todo)

# piezas con respuesta, en train fuera de fold; en el mismo orden que en los notebooks
claves = {"te_seller_id": todo["seller_id"], "te_category_id": todo["category_id"], **claves_cruzadas(todo)}
texto = texto_combinado(todo)
piezas = {}
for nombre in ["te_seller_id", "te_category_id", "prob_combinado", *claves_cruzadas(todo)]:
    if nombre == "prob_combinado":
        piezas[nombre] = prob_texto_fuera_de_fold(train(texto), y, caracteres=True)
    else:
        piezas[nombre] = target_encoding(train(claves[nombre]), y, train(claves[nombre]).iloc[:0])[0]
piezas_nb = {**piezas, "prob_nbsvm": prob_nbsvm_fuera_de_fold(M[:N_TRAIN], y)}

# lo que se aplica a test: tablas de target encoding y modelos de texto, ajustados con todo train
prior = y.mean()
tablas = {n: tabla_target_encoding(train(c), y, prior) for n, c in claves.items()}
texto_modelo = modelo_texto(caracteres=True).fit(train(texto), y)
nbsvm = ajustar_nbsvm(M[:N_TRAIN], y)

# los cuatro modelos: (features, piezas, modelo)
MODELOS = [
    (X_todo, piezas_nb, LGBMClassifier(**PARAMS_LGBM)),
    (X_svd, piezas_nb, LGBMClassifier(**PARAMS_LGBM)),
    (X_svd, piezas, CatBoostClassifier(**PARAMS_CAT)),
    (X_svd, piezas, LGBMClassifier(**PARAMS_DART)),
]
modelos, columnas = [], []
for X, pz, modelo in MODELOS:
    A = X.iloc[:N_TRAIN].reset_index(drop=True)
    for nombre, valores in pz.items():
        A[nombre] = valores
    categorias = {c: pd.Categorical(A[c]).categories for c in CAT}
    A = fijar_categorias(A, categorias)
    if isinstance(modelo, CatBoostClassifier):
        modelo.fit(preparar_catboost(A), y, cat_features=CAT)
    else:
        modelo.fit(A, y)
    modelos.append(modelo)
    columnas.append(list(A.columns))
    print(f"{type(modelo).__name__} entrenado con {A.shape[1]} columnas", flush=True)

os.makedirs("modelo", exist_ok=True)
with open("modelo/modelo_final.pkl", "wb") as f:
    pickle.dump({"prior": prior, "tablas": tablas, "texto_modelo": texto_modelo, "nbsvm": nbsvm,
                 "categorias": categorias, "modelos": modelos, "columnas": columnas}, f)
print("modelo guardado en modelo/modelo_final.pkl")

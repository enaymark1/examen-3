# ¿Nuevo o usado? · Examen 3 de Análisis Predictivo (ITBA)

Modelo para predecir si una publicación de MercadoLibre es de un producto **nuevo o usado**, a partir de los datos de la publicación (título, precio, fotos, envío, medios de pago, vendedor, etc.).

- **Competencia:** [ap-2026-q-2 en Kaggle](https://www.kaggle.com/competitions/ap-2026-q-2)
- **Métrica:** accuracy. Las clases están casi balanceadas: 53,7% nuevos y 46,3% usados.
- **Datos:** 70.000 publicaciones de entrenamiento (con la respuesta) y 30.000 de test (sin ella), en JSON anidado.
- **Autor:** Ezequiel Naymark

## Entrega

| Consigna | Archivo |
|---|---|
| a. Presentación | *(pendiente)* |
| b. Entrena y guarda el modelo final | [`train.py`](train.py) → guarda `modelo/modelo_final.pkl` |
| c. Aplica el modelo a test y reproduce el archivo enviado | [`predict.py`](predict.py) → genera [`submissions/08_ensamble_4_nbsvm_cat_dart.csv`](submissions/08_ensamble_4_nbsvm_cat_dart.csv) |

Los dos scripts importan las funciones de features de [`features.py`](features.py). Para correrlos, ver [Cómo reproducir](#cómo-reproducir).

Los notebooks `01` a `08` no son parte de la entrega obligatoria: documentan el desarrollo (qué se probó, qué funcionó y qué no) y respaldan las decisiones del modelo final.

## Resultado

| Modelo | Accuracy en validación cruzada | Kaggle (público) |
|---|---|---|
| Baseline: LightGBM por defecto, 16 features | 0,8417 | 0,83814 |
| **Modelo final (ensamble 08)** | **0,9462** | **0,94671** |

La validación cruzada (5 particiones) anticipó bien el resultado de Kaggle en todas las submissions, así que las decisiones se tomaron con ella y no con el leaderboard público.

## Cómo se llegó

Cada paso está en un notebook numerado, con explicaciones y resultados:

| Notebook | Qué se hizo | Validación | Kaggle |
|---|---|---|---|
| [`01_baseline.ipynb`](01_baseline.ipynb) | Carga de datos, 16 features directas del JSON, regresión logística y LightGBM | 0,8417 | 0,83814 |
| [`02_features.ipynb`](02_features.ipynb) | 43 features más del JSON, *target encoding* de vendedor y categoría, probabilidad de usado según el título (TF-IDF + regresión logística) | 0,9376 | 0,93809 |
| [`03_fotos_vendedor_texto.ipynb`](03_fotos_vendedor_texto.ipynb) | Features de las fotos (proporción, antigüedad), perfil del vendedor, texto combinado (título + garantía + atributos), hiperparámetros, importancia de features | 0,9423 | 0,94457 |
| [`04_modelos_ensamble.ipynb`](04_modelos_ensamble.ipynb) | XGBoost y CatBoost, ensamble, análisis por tamaño del vendedor, variantes del modelo de texto | 0,9433 | 0,94466 |
| [`05_errores_encodings.ipynb`](05_errores_encodings.ipynb) | Análisis de errores, *target encodings* cruzados (vendedor × tipo de publicación, etc.) y quince ideas que no funcionaron | 0,9443 | 0,94590 |
| [`06_svd_modelos_stacking.ipynb`](06_svd_modelos_stacking.ipynb) | SVD del texto, más modelos (CatBoost, Random Forest, Extra Trees, lineal), stacking, reglas a mano | 0,9457 | 0,94614 |
| [`07_embeddings_variantes.ipynb`](07_embeddings_variantes.ipynb) | Embeddings de un modelo de lenguaje preentrenado; variantes de LightGBM (`extra_trees`, DART) y CatBoost | 0,9457 | — |
| [`08_nbsvm_e5_ensambles.ipynb`](08_nbsvm_e5_ensambles.ipynb) | NB-SVM (segunda probabilidad del texto), ensambles con CatBoost, DART, XGBoost y `extra_trees`, embeddings e5, pseudo-etiquetado, pesos *greedy*; submissions 07 a 11 | 0,9462 | 0,94671 |

Los notebooks documentan **también lo que no funcionó** y por qué, para que se vea cómo se eligió cada parte del modelo.

## Modelo final

**Promedio simple de cuatro modelos de boosting** (ensamble 08). Todos parten de las mismas **98 columnas**:

- **92 features** que no usan la respuesta: precio, cantidades, tipo de publicación, envío, medios de pago, tags, fechas, palabras clave del título y la garantía, tamaño y antigüedad de las fotos, y el perfil de cómo publica cada vendedor.
- **6 features que sí usan la respuesta**, siempre calculadas fuera de fold:
  - *target encoding* de vendedor, categoría, vendedor × categoría, vendedor × tipo de publicación y categoría × primera palabra del título;
  - probabilidad de usado según el texto (TF-IDF de palabras y caracteres + regresión logística).

Sobre esa base, cada modelo ve el texto de otra forma:

| Modelo | Columnas extra | Hiperparámetros |
|---|---|---|
| **LightGBM A** | probabilidad NB-SVM del texto | 1.500 árboles, `learning_rate=0.02`, `num_leaves=63`, `colsample_bytree=0.5` |
| **LightGBM B** | NB-SVM + 128 columnas de SVD (el TF-IDF del texto resumido en sus 128 patrones principales) | los mismos |
| **CatBoost** | 128 columnas de SVD | 2.000 iteraciones, `learning_rate=0.06`, `depth=7` |
| **LightGBM DART** | 128 columnas de SVD | 800 árboles, `learning_rate=0.08`, `drop_rate=0.1` |

**NB-SVM** es una segunda probabilidad del texto: cada palabra se pondera por su "razón de usados" (cuánto más aparece en usados que en nuevos) y sobre eso se ajusta una regresión logística. También se calcula fuera de fold.

Se predice "usado" si el promedio de las cuatro probabilidades es mayor a 0,5.

Cada modelo solo da casi lo mismo, pero **se equivocan en publicaciones distintas**, así que el promedio mejora. Con más modelos (8 o 10, incluyendo XGBoost y embeddings de un modelo de lenguaje) la validación ya no mejora, así que se eligió el de 4.

Las features más importantes son la probabilidad del texto y el historial del vendedor.

## Validación

- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`. Train y test cubren el mismo período, así que la partición es aleatoria y no temporal.
- **Sin *leakage*:** todo lo que usa la respuesta (*target encodings*, modelo de texto) se recalcula **adentro de cada partición**, solo con las filas de entrenamiento. En train se calcula además fuera de fold, en 5 sub-particiones.
- El ruido entre particiones es de ~±0,001. Una idea se aceptó solo si mejoraba más que eso y **en casi todas las particiones**.
- Todo usa `random_state=42` y es reproducible: cada notebook regenera exactamente el CSV que se envió a Kaggle.

## Limitaciones

- **Vendedores con una sola publicación** (25% de los datos): sin historial, el error es de ~11%, contra ~2% en vendedores con más de 50 publicaciones. Juntan casi la mitad de los errores.
- **Ruido en la etiqueta:** la condición la declara el vendedor. Un tercio de los errores son casos donde la publicación contradice su etiqueta, como libros que dicen "Usado" marcados como nuevos o ebooks marcados como usados. Ningún modelo puede acertarlos.

## Estructura del repositorio

```
├── 01_baseline.ipynb … 08_nbsvm_e5_ensambles.ipynb   notebooks de cada etapa
├── features.py        funciones de features compartidas (las usan los notebooks desde el 04 y los scripts)
├── train.py           entrena el modelo final y lo guarda en modelo/ (no está en el repo: se regenera)
├── predict.py         aplica el modelo guardado a test y genera la submission
├── submissions/       los CSV enviados a Kaggle, con nombres que dicen qué modelo tenían
└── data/              los datos de Kaggle (no están en el repo, ver abajo)
```

## Cómo reproducir

1. Descargar `train_data.jsonlines` y `test_data.jsonlines` de la [página de datos de la competencia](https://www.kaggle.com/competitions/ap-2026-q-2/data) y ponerlos en `data/`. No están en el repo porque superan el límite de tamaño de GitHub.
2. Instalar las librerías. Se usaron estas versiones:

   | Python | numpy | scipy | pandas | scikit-learn | lightgbm | xgboost | catboost | matplotlib |
   |---|---|---|---|---|---|---|---|---|
   | 3.13.2 | 2.3.4 | 1.18.0 | 3.0.1 | 1.9.0 | 4.7.0 | 3.4.1 | 1.2.10 | 3.10.7 |

   Los notebooks 07 y 08 usan además `torch` 2.14.1 (CPU) y `sentence-transformers` 6.2.0 para los embeddings. El modelo final no los necesita.

3. Para el modelo final: `python train.py` (~1 hora) y después `python predict.py` (~10 minutos), desde la carpeta del repo. Regeneran exactamente `submissions/08_ensamble_4_nbsvm_cat_dart.csv`.
4. Los notebooks también se pueden correr desde la carpeta del repo. Cada uno es independiente y regenera sus submissions en `submissions/`. Los últimos tardan entre 1,5 y 5 horas, porque cada experimento se valida en 5 particiones.

# ¿Nuevo o usado? · Examen 3 de Análisis Predictivo (ITBA)

Modelo para predecir si una publicación de MercadoLibre es de un producto **nuevo o usado**, a partir de los datos de la publicación (título, precio, fotos, envío, medios de pago, vendedor, etc.).

- **Competencia:** [ap-2026-q-2 en Kaggle](https://www.kaggle.com/competitions/ap-2026-q-2)
- **Métrica:** accuracy. Las clases están casi balanceadas: 53,7% nuevos y 46,3% usados.
- **Datos:** 70.000 publicaciones de entrenamiento (con la respuesta) y 30.000 de test (sin ella), en JSON anidado.
- **Autor:** Ezequiel Naymark

## Resultado

| Modelo | Accuracy en validación cruzada | Kaggle (público) |
|---|---|---|
| Baseline: LightGBM por defecto, 16 features | 0,8417 | 0,83814 |
| **Modelo final (06)** | **0,9457** | **0,94614** |

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

Los notebooks documentan **también lo que no funcionó** y por qué, para que se vea cómo se eligió cada parte del modelo.

## Modelo final

**Promedio de dos LightGBM** (1.500 árboles, `learning_rate=0.02`, `num_leaves=63`, `colsample_bytree=0.5`) que ven el texto de dos formas distintas:

1. **LightGBM A**, con 98 columnas:
   - **92 features** que no usan la respuesta: precio, cantidades, tipo de publicación, envío, medios de pago, tags, fechas, palabras clave del título y la garantía, tamaño y antigüedad de las fotos, y el perfil de cómo publica cada vendedor.
   - **6 features que sí usan la respuesta**, siempre calculadas fuera de fold:
     - *target encoding* de vendedor, categoría, vendedor × categoría, vendedor × tipo de publicación y categoría × primera palabra del título;
     - probabilidad de usado según el texto (TF-IDF de palabras y caracteres + regresión logística).
2. **LightGBM B**, con las mismas 98 columnas **más 128 columnas de SVD**: el TF-IDF del texto resumido en sus 128 patrones principales.

Se predice "usado" si el promedio de las dos probabilidades es mayor a 0,5.

Cada uno solo da casi lo mismo, pero **se equivocan en publicaciones distintas**, así que el promedio mejora en las 5 particiones de la validación.

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
├── 01_baseline.ipynb … 06_svd_modelos_stacking.ipynb   notebooks de cada etapa
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

3. Para el modelo final: `python train.py` y después `python predict.py` (desde la carpeta del repo). Reproducen exactamente el CSV enviado.
4. Los notebooks también se pueden correr desde la carpeta del repo. Cada uno es independiente y regenera su submission en `submissions/`. Los últimos tardan entre 1,5 y 2 horas, porque cada experimento se valida en 5 particiones.

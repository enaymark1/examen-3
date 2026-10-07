# Examen 3 · Análisis Predictivo (ITBA) — Competencia Kaggle

## Contexto

- Competencia: https://www.kaggle.com/competitions/ap-2026-q-2
- Repo: https://github.com/enaymark1/examen-3
- Tarea: predecir si un ítem de MercadoLibre es **nuevo o usado** (`condition` = `new` / `used`).
- Métrica: **accuracy**. Clases casi balanceadas (53,7% new / 46,3% used).
- **Examen individual.** La nota depende mucho del score del leaderboard; los mejores de la comisión rondan 0,93.
- Límite: **5 submissions por día** en Kaggle.

## Qué hay que entregar (consigna)

1. **Presentación de máximo 5 minutos** (PDF o PPT, formato ITBA) con:
   1. Modelo baseline: features y rendimiento.
   2. Selección de modelos: qué se probó (modelos, features, encodings, hiperparámetros), cómo y por qué.
   3. Modelo final: cómo se llegó, qué features usa, rendimiento.
   4. Limitaciones y posibles mejoras.
   El objetivo es demostrar que **el resultado no es casualidad ni copiado**: hay que entender todo el código.
2. **`train.py`** (o notebook): entrena y guarda el modelo final. Tiene que replicarlo **perfectamente** y no incluir código que no sirva para eso.
3. **`predict.py`** (o notebook): aplica el modelo guardado a test y reproduce **exactamente** el CSV enviado a Kaggle. Sin código extra.

## Datos

- `data/train_data.jsonlines` (70.000 ítems, 45 campos, 200 MB) y `data/test_data.jsonlines` (30.000, 44 campos: sin `condition`).
- JSON anidado: `shipping`, `seller_address`, `pictures`, `attributes`, `non_mercado_pago_payment_methods`, etc.
- `data/` está en `.gitignore`: GitHub rechaza archivos de más de 100 MB. **Nunca commitear los datos.**
- Cargar con `pd.read_json(..., lines=True, keep_default_dates=False)`. Sin ese parámetro, pandas convierte solo las columnas que terminan en `_time` a fechas y rompe los cálculos.

## Formato de la submission (confirmado probando)

```
ID,condition
1,used
2,new
```

- `ID` = número de fila de test **empezando en 1** (`range(1, len(test) + 1)`), en el orden del archivo.
- No es el `id` tipo `MLA...` ni el índice desde 0: los dos dieron error.

## Validación

- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- Train y test cubren el mismo período (2013-05 a 2015-10): partición aleatoria, no temporal.
- Vendedores de validación presentes en train: 69% en la CV, 72% en el test real. La CV imita bien al test.
- Ruido entre particiones: ~±0,001. Mejoras menores a eso no justifican complicar el modelo.
- **La validación local predice el leaderboard:**

| Submission | CV | Kaggle |
|---|---|---|
| `01_baseline_lgbm.csv` | 0,8417 | 0,83814 |
| `02_vendedor_titulo.csv` | 0,9376 | 0,93809 |
| `03_fotos_vendedor_texto.csv` | 0,9423 | (pendiente) |

## Resultados hasta ahora

| Experimento | Accuracy CV |
|---|---|
| Clase mayoritaria | 0,5370 |
| Regresión logística, 16 features | 0,8229 |
| LightGBM por defecto (100 árboles), 16 features | 0,8417 |
| LightGBM 500 árboles, 16 features | 0,8471 |
| + 43 features extra del JSON | 0,9057 |
| + target encoding de vendedor y categoría | 0,9322 |
| + probabilidad del título (TF-IDF + logística) | 0,9376 (0,9372 al re-correr en el 03) |
| + fotos (tamaño, proporción, mes de subida) | 0,9388 |
| + fotos, otros campos y perfil del vendedor | 0,9399 |
| + texto combinado (título + garantía + atributos), en lugar del título | 0,9416 |
| + hiperparámetros (1500 árboles, lr 0,02, colsample 0,5) | **0,9423** |

Probado y descartado en el 03: target encoding de la ciudad, TF-IDF de caracteres del título, modelo de texto solo de la garantía (ninguno mejora más que el ruido).

## Decisiones del notebook 02

- **Baseline para la presentación:** LightGBM por defecto con 16 features (0,8417).
- **Target encoding** de `seller_id` y `category_id`: proporción de usados con suavizado 10 hacia el promedio general. En train se calcula **fuera de fold** (5 particiones); en test, con todo train. En la validación se **recalcula adentro de cada partición** con solo los datos de entrenamiento, para no inflar la nota.
- **Título:** `TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)` + `LogisticRegression(C=4)`. Su probabilidad entra como feature, también fuera de fold.
- Conteos de ítems por vendedor y por categoría calculados sobre train + test (no usan la respuesta).
- Categóricas de LightGBM: alinear las categorías de validación/test con las de entrenamiento (`alinear_categorias`).
- Todo con `random_state=42` para que sea reproducible.

## Decisiones del notebook 03 (`03_fotos_vendedor_texto.ipynb`, modelo actual)

- Features nuevas sobre train + test (no usan la respuesta): fotos (`features_fotos`), otros campos (`features_otros`), perfil del vendedor (`features_vendedor`). `status` pasa a ser categórica. Total: 92 columnas + 3 piezas.
- Fotos: 4:3 (celular) → usado; cuadradas (catálogo) → nuevo. El `id` de la foto termina en `_MMAAAA` (mes de subida).
- Un solo modelo de texto: `prob_combinado` = TF-IDF + logística sobre `título || garantía || atributos`. Reemplaza a `prob_titulo`.
- Piezas que usan la respuesta (target encoding y texto) se calculan una vez por partición y se reutilizan (`piezas`); la partición 5 es train → test.
- **`cross_val_predict` sin `n_jobs`:** paralelizarlo cambia levemente la logística y ~250 predicciones finales. Sin paralelizar es reproducible.
- Parámetros finales: `n_estimators=1500, learning_rate=0.02, num_leaves=63, colsample_bytree=0.5, random_state=42`. La fila 15 (num_leaves=127 + subsample) dio 0,9429, pero +0,0006 es ruido y es más compleja.
- Importancia (gain): `prob_combinado` 29%, `te_seller_id` 20%, `listing_type_id` 9%, `initial_quantity` 6%; las 20 primeras suman 87%.
- El notebook tarda ~37 min en correr entero (15 min las piezas).

## Próximos pasos

1. Mandar `03_fotos_vendedor_texto.csv` a Kaggle y anotar el score.
2. Probar otros modelos (por ejemplo CatBoost o XGBoost, no están instalados) y, si conviene, combinarlos.
3. Armar `train.py` y `predict.py` limpios y verificar que reproducen exactamente la última submission.
4. Presentación.

## Cómo trabajar con Ezequiel

- Escribir en castellano rioplatense.
- **No maneja bien Git ni Kaggle.** Al final de cada paso, decirle explícitamente si tiene que hacer algo en Git y en Kaggle, y cómo, paso a paso. Si no hace falta nada, aclararlo.
- Explicar el código lo suficiente como para que lo pueda defender en el examen. Antes de cada cambio grande, decir qué se va a hacer y por qué.
- Validar cada paso antes de pasar al siguiente, y mandar a Kaggle solo lo que ya mejoró en la CV (hay 5 envíos por día).
- Guardar cada submission en `submissions/` con un nombre que diga qué tenía, y anotar el score de CV y de Kaggle en la tabla de arriba.
- Todo lo que se prueba va a un notebook numerado dentro de `examen-3/`, explicado como el 02. No dejar nada en carpetas temporales.

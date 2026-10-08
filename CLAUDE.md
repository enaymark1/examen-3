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
| `03_fotos_vendedor_texto.csv` | 0,9423 | 0,94457 |
| `04_lgbm_xgb_texto_v4.csv` | 0,9433 | **0,94466** |

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
| + hiperparámetros (1500 árboles, lr 0,02, colsample 0,5) | 0,9423 |
| XGBoost solo / CatBoost solo | 0,9428 / 0,9414 |
| Promedio LightGBM + XGBoost | 0,9429 |
| Texto v4 (palabras + caracteres) + promedio LightGBM + XGBoost | **0,9433** (mejora en 5/5 particiones) |

Probado y descartado en el 03: target encoding de la ciudad, TF-IDF de caracteres del título, modelo de texto solo de la garantía (ninguno mejora más que el ruido).

Probado y descartado en el 04: CatBoost (más lento, no suma al promedio), texto con tokens estructurados (el texto solo sube a 0,906 pero el modelo baja a 0,9415: repite info que ya está en columnas), `C=2` / `C=8`, ajustar el umbral (+0,0002, ruido). También se miró: títulos repetidos train/test (1,5%) y `parent_item_id` (no apunta a train): no vale la pena.

## Decisiones del notebook 02

- **Baseline para la presentación:** LightGBM por defecto con 16 features (0,8417).
- **Target encoding** de `seller_id` y `category_id`: proporción de usados con suavizado 10 hacia el promedio general. En train se calcula **fuera de fold** (5 particiones); en test, con todo train. En la validación se **recalcula adentro de cada partición** con solo los datos de entrenamiento, para no inflar la nota.
- **Título:** `TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)` + `LogisticRegression(C=4)`. Su probabilidad entra como feature, también fuera de fold.
- Conteos de ítems por vendedor y por categoría calculados sobre train + test (no usan la respuesta).
- Categóricas de LightGBM: alinear las categorías de validación/test con las de entrenamiento (`alinear_categorias`).
- Todo con `random_state=42` para que sea reproducible.

## Decisiones del notebook 03 (`03_fotos_vendedor_texto.ipynb`)

- Features nuevas sobre train + test (no usan la respuesta): fotos (`features_fotos`), otros campos (`features_otros`), perfil del vendedor (`features_vendedor`). `status` pasa a ser categórica. Total: 92 columnas + 3 piezas.
- Fotos: 4:3 (celular) → usado; cuadradas (catálogo) → nuevo. El `id` de la foto termina en `_MMAAAA` (mes de subida).
- Un solo modelo de texto: `prob_combinado` = TF-IDF + logística sobre `título || garantía || atributos`. Reemplaza a `prob_titulo`.
- Piezas que usan la respuesta (target encoding y texto) se calculan una vez por partición y se reutilizan (`piezas`); la partición 5 es train → test.
- **`cross_val_predict` sin `n_jobs`:** paralelizarlo cambia levemente la logística y ~250 predicciones finales. Sin paralelizar es reproducible.
- Parámetros finales: `n_estimators=1500, learning_rate=0.02, num_leaves=63, colsample_bytree=0.5, random_state=42`. La fila 15 (num_leaves=127 + subsample) dio 0,9429, pero +0,0006 es ruido y es más compleja.
- Importancia (gain): `prob_combinado` 29%, `te_seller_id` 20%, `listing_type_id` 9%, `initial_quantity` 6%; las 20 primeras suman 87%.
- El notebook tarda ~37 min en correr entero (15 min las piezas).

## Decisiones del notebook 04 (`04_modelos_ensamble.ipynb`, modelo actual)

- Las funciones de features están en **`features.py`** (verificado: reproduce exacto el CSV 03). Notebooks desde el 04 y `train.py` / `predict.py` importan de ahí.
- Texto v4: `calcular_piezas(..., caracteres=True)` → TF-IDF de palabras (1-2) + caracteres `char_wb` (3-5, min_df=5, 200.000 máx.) unidos, `LogisticRegression(C=4)`.
- Modelo final: promedio de probabilidades de LightGBM (`PARAMS_LGBM`, los del 03) y XGBoost (`n_estimators=1500, learning_rate=0.02, max_depth=8, subsample=0.8, colsample_bytree=0.5, min_child_weight=2, tree_method="hist", enable_categorical=True, random_state=42, n_jobs=8`). Umbral 0,5.
- Los 3 modelos coinciden en 98,5–99% de los ítems (correlación ~0,995): con estas features el algoritmo importa poco.
- **Limitación principal:** vendedores con 1 solo ítem (25% de los datos) tienen error 11,2% vs 2,1% en vendedores con más de 50; juntan casi la mitad de los errores.
- La 04 cambia 318 predicciones respecto de la 03. Kaggle: +0,00009, o sea, prácticamente igual. Estamos en una meseta de ~0,943–0,945.
- El notebook tarda ~90 min. CatBoost deja una carpeta `catboost_info/` (está en `.gitignore`).
- Notebooks largos: que la compu no se suspenda (apagar la pantalla no importa).

## Próximos pasos

1. Armar `train.py` y `predict.py` limpios y verificar que reproducen exactamente la última submission.
2. Presentación.

## Cómo trabajar con Ezequiel

- Escribir en castellano rioplatense.
- **No maneja bien Git ni Kaggle.** Al final de cada paso, decirle explícitamente si tiene que hacer algo en Git y en Kaggle, y cómo, paso a paso. Si no hace falta nada, aclararlo.
- Explicar el código lo suficiente como para que lo pueda defender en el examen. Antes de cada cambio grande, decir qué se va a hacer y por qué.
- Validar cada paso antes de pasar al siguiente, y mandar a Kaggle solo lo que ya mejoró en la CV (hay 5 envíos por día).
- Guardar cada submission en `submissions/` con un nombre que diga qué tenía, y anotar el score de CV y de Kaggle en la tabla de arriba.
- Todo lo que se prueba va a un notebook numerado dentro de `examen-3/`, explicado como el 02. No dejar nada en carpetas temporales.

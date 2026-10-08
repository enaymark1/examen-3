"""Features del modelo, en un solo lugar.

Es el código de los notebooks 02 y 03, sin cambios en la lógica. Lo usan los
notebooks desde el 04 y los scripts finales (train.py / predict.py), así lo que
se valida es exactamente lo que se entrega.
"""
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline, make_union

CAT = ["listing_type_id", "buying_mode", "shipping_mode", "state", "status"]


def cargar_datos(carpeta="data"):
    # keep_default_dates=False: que pandas NO convierta solo las columnas que terminan en "_time"
    train = pd.read_json(f"{carpeta}/train_data.jsonlines", lines=True, keep_default_dates=False)
    test = pd.read_json(f"{carpeta}/test_data.jsonlines", lines=True, keep_default_dates=False)
    y = (train["condition"] == "used").astype(int).values  # 1 = usado, 0 = nuevo
    todo = pd.concat([train.drop(columns="condition"), test], ignore_index=True)
    return todo, y


def g(d, k):
    # lee una clave de un diccionario sin romper si el campo viene vacío
    return d.get(k) if isinstance(d, dict) else None


def dimensiones(s):
    # "1200x900" -> (1200, 900); si viene mal formado, (nan, nan)
    try:
        ancho, alto = s.split("x")
        return int(ancho), int(alto)
    except Exception:
        return np.nan, np.nan


# ---------- 02: las 16 del baseline ----------

def armar_features(df):
    X = pd.DataFrame(index=df.index)
    for c in ["price", "base_price", "initial_quantity", "sold_quantity", "available_quantity"]:
        X[c] = df[c]
    X["accepts_mercadopago"] = df["accepts_mercadopago"].astype(int)
    X["automatic_relist"] = df["automatic_relist"].astype(int)
    X["free_shipping"] = df["shipping"].apply(lambda s: int(bool(g(s, "free_shipping"))))
    X["local_pick_up"] = df["shipping"].apply(lambda s: int(bool(g(s, "local_pick_up"))))
    X["n_pictures"] = df["pictures"].apply(len)
    X["n_variations"] = df["variations"].apply(len)
    X["n_attributes"] = df["attributes"].apply(len)
    X["has_warranty"] = df["warranty"].notna().astype(int)
    X["listing_type_id"] = df["listing_type_id"]
    X["buying_mode"] = df["buying_mode"]
    X["shipping_mode"] = df["shipping"].apply(lambda s: g(s, "mode"))
    return X


# ---------- 02: las 43 extra ----------

def armar_features_extra(df):
    X = pd.DataFrame(index=df.index)
    X["n_pagos"] = df["non_mercado_pago_payment_methods"].apply(len)
    pagos = df["non_mercado_pago_payment_methods"].apply(lambda L: {d.get("id") for d in L})
    for p in ["MLATB", "MLAWC", "MLAMO", "MLAOT", "MLAMC", "MLAVE", "MLAMS"]:
        X["pago_" + p] = pagos.apply(lambda s: int(p in s))
    X["n_tags"] = df["tags"].apply(len)
    for t in ["dragged_bids_and_visits", "good_quality_thumbnail", "dragged_visits",
              "free_relist", "poor_quality_thumbnail"]:
        X["tag_" + t] = df["tags"].apply(lambda L: int(t in L))
    X["tiene_sub_status"] = df["sub_status"].apply(len)
    X["tiene_video"] = df["video_id"].notna().astype(int)
    X["tiene_tienda_oficial"] = df["official_store_id"].notna().astype(int)
    X["tiene_precio_original"] = df["original_price"].notna().astype(int)
    X["tiene_parent"] = df["parent_item_id"].notna().astype(int)
    X["n_descripciones"] = df["descriptions"].apply(len)
    X["dif_precio_base"] = df["price"] - df["base_price"]
    X["precio_redondo"] = (df["price"] % 10 == 0).astype(int)
    X["duracion_dias"] = (df["stop_time"] - df["start_time"]) / 86_400_000  # vienen en milisegundos
    creado = pd.to_datetime(df["date_created"], utc=True)
    actualizado = pd.to_datetime(df["last_updated"], utc=True)
    X["dias_desde_2013"] = (creado - pd.Timestamp("2013-01-01", tz="UTC")).dt.total_seconds() / 86400
    X["horas_hasta_update"] = (actualizado - creado).dt.total_seconds() / 3600
    titulo = df["title"].str.lower()
    X["largo_titulo"] = df["title"].str.len()
    for w in ["usad", "nuev", "original", "impecable", "excelente", "estado",
              "garant", "sin uso", "oferta", "import"]:
        X["titulo_" + w.replace(" ", "_")] = titulo.str.contains(w).astype(int)
    garantia = df["warranty"].fillna("").str.lower()
    X["largo_garantia"] = garantia.str.len()
    X["garantia_sin"] = garantia.str.contains("sin garant").astype(int)
    X["garantia_nuevo"] = garantia.str.contains("nuev").astype(int)
    X["garantia_fabrica"] = garantia.str.contains("fabric").astype(int)
    X["state"] = df["seller_address"].apply(lambda a: g(g(a, "state"), "name"))
    # conteos sobre train + test: df tiene que ser `todo`
    X["vendedor_n_items"] = df["seller_id"].map(df["seller_id"].value_counts()).values
    X["categoria_n_items"] = df["category_id"].map(df["category_id"].value_counts()).values
    return X


# ---------- 03: fotos ----------

def features_fotos(df):
    X = pd.DataFrame(index=df.index)
    primera = df["pictures"].apply(lambda L: dimensiones(L[0]["max_size"]) if L else (np.nan, np.nan))
    X["foto_ancho"] = primera.str[0]
    X["foto_alto"] = primera.str[1]
    X["foto_aspecto"] = X["foto_ancho"] / X["foto_alto"]
    X["foto_area"] = X["foto_ancho"] * X["foto_alto"]

    todas = df["pictures"].apply(lambda L: [dimensiones(p["max_size"]) for p in L])
    X["fotos_cuadradas"] = todas.apply(lambda L: np.mean([w == h for w, h in L]) if L else np.nan)
    X["fotos_4_3"] = todas.apply(lambda L: np.mean([max(w, h) * 3 == min(w, h) * 4 for w, h in L]) if L else np.nan)
    X["fotos_vertical"] = todas.apply(lambda L: np.mean([h > w for w, h in L]) if L else np.nan)
    X["fotos_area_max"] = todas.apply(lambda L: max(w * h for w, h in L) if L else np.nan)

    # mes de subida de cada foto, a partir del final del id: _MMAAAA
    def meses_fotos(L):
        meses = []
        for p in L:
            m = re.search(r"_(\d{2})(\d{4})$", p["id"])
            if m:
                meses.append(int(m.group(2)) * 12 + int(m.group(1)))
        return meses
    meses = df["pictures"].apply(meses_fotos)
    creado = pd.to_datetime(df["date_created"], utc=True)
    X["foto_meses_antes"] = (creado.dt.year * 12 + creado.dt.month) - meses.apply(lambda L: min(L) if L else np.nan)
    X["foto_tiene_fecha"] = meses.apply(lambda L: int(len(L) > 0))
    X["fotos_distintas"] = df["pictures"].apply(lambda L: len({p["id"] for p in L}))
    return X


# ---------- 03: otros campos ----------

def features_otros(df):
    X = pd.DataFrame(index=df.index)
    X["status"] = df["status"]
    X["usd"] = (df["currency_id"] == "USD").astype(int)
    X["n_deals"] = df["deal_ids"].apply(len)
    X["tiene_catalogo"] = df["catalog_product_id"].notna().astype(int)
    X["log_precio"] = np.log1p(df["price"])
    mediana_categoria = np.log1p(df["price"]).groupby(df["category_id"]).transform("median")
    X["precio_vs_categoria"] = X["log_precio"] - mediana_categoria
    X["vendidos_sobre_inicial"] = df["sold_quantity"] / df["initial_quantity"].clip(lower=1)
    X["cant_inicial_1"] = (df["initial_quantity"] == 1).astype(int)
    X["n_attr_variaciones"] = df["variations"].apply(lambda L: sum(len(v.get("attribute_combinations") or []) for v in L))
    creado = pd.to_datetime(df["date_created"], utc=True)
    X["hora_creado"] = creado.dt.hour
    X["dia_semana"] = creado.dt.dayofweek
    return X


# ---------- 03: perfil del vendedor (sin usar la respuesta) ----------

def features_vendedor(df):
    X = pd.DataFrame(index=df.index)
    vendedor = df["seller_id"]
    log_precio = np.log1p(df["price"])
    primera_cuadrada = df["pictures"].apply(lambda L: float(np.equal(*dimensiones(L[0]["max_size"]))) if L else np.nan)
    ciudad = df["seller_address"].apply(lambda a: g(g(a, "city"), "name"))
    creado = pd.to_datetime(df["date_created"], utc=True)

    X["vend_precio_medio"] = log_precio.groupby(vendedor).transform("mean")
    X["vend_precio_std"] = log_precio.groupby(vendedor).transform("std")
    X["vend_n_categorias"] = df.groupby("seller_id")["category_id"].transform("nunique")
    X["vend_cant_inicial_media"] = df["initial_quantity"].groupby(vendedor).transform("mean")
    X["vend_prop_gold"] = df["listing_type_id"].str.startswith("gold").groupby(vendedor).transform("mean")
    X["vend_prop_garantia"] = df["warranty"].notna().groupby(vendedor).transform("mean")
    X["vend_n_fotos_media"] = df["pictures"].apply(len).groupby(vendedor).transform("mean")
    X["vend_prop_foto_cuadrada"] = primera_cuadrada.groupby(vendedor).transform("mean")
    X["vend_n_ciudades"] = ciudad.groupby(vendedor).transform("nunique")
    X["vend_dias_actividad"] = creado.groupby(vendedor).transform(lambda t: (t.max() - t.min()).days)
    X["vend_prop_paused"] = (df["status"] == "paused").groupby(vendedor).transform("mean")
    return X


def armar_tabla(todo):
    """Las 92 features que no usan la respuesta, para train + test juntos."""
    return pd.concat([armar_features(todo), armar_features_extra(todo), features_fotos(todo),
                      features_otros(todo), features_vendedor(todo)], axis=1)


def texto_combinado(todo):
    garantia = todo["warranty"].fillna("sin_dato").astype(str)
    atributos = todo["attributes"].apply(lambda L: " ".join(f"{a.get('id')}={a.get('value_name')}" for a in L))
    return todo["title"] + " || " + garantia + " || " + atributos


# ---------- piezas que usan la respuesta ----------

def target_encoding(clave_train, y_train, clave_test, suavizado=10):
    """Proporción de usados por clave, suavizada hacia el promedio. Fuera de fold en train."""
    prior = y_train.mean()

    def tabla_proporciones(claves, ys):
        t = pd.DataFrame({"k": np.asarray(claves), "y": ys}).groupby("k")["y"].agg(["sum", "count"])
        return (t["sum"] + suavizado * prior) / (t["count"] + suavizado)

    enc_train = np.full(len(clave_train), prior)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for idx_a, idx_b in kf.split(clave_train, y_train):
        tabla = tabla_proporciones(clave_train.iloc[idx_a], y_train[idx_a])
        enc_train[idx_b] = clave_train.iloc[idx_b].map(tabla).fillna(prior).values
    tabla = tabla_proporciones(clave_train, y_train)
    enc_test = clave_test.map(tabla).fillna(prior).values
    return enc_train, enc_test


def vectorizador_texto(caracteres=False):
    """TF-IDF de palabras (notebook 03). Con caracteres=True, además pedazos de 3 a 5 letras (notebook 04)."""
    palabras = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    if not caracteres:
        return palabras
    letras = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, sublinear_tf=True, max_features=200_000)
    return make_union(palabras, letras)


def prob_texto(textos_train, y_train, textos_test, caracteres=False):
    """Probabilidad de usado según el texto (TF-IDF + logística). Fuera de fold en train.
    Sin n_jobs: paralelizar cambia levemente el resultado."""
    modelo = make_pipeline(vectorizador_texto(caracteres), LogisticRegression(max_iter=3000, C=4))
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    p_train = cross_val_predict(modelo, textos_train, y_train, cv=kf, method="predict_proba")[:, 1]
    modelo.fit(textos_train, y_train)
    p_test = modelo.predict_proba(textos_test)[:, 1]
    return p_train, p_test


def calcular_piezas(todo, y, idx_a, idx_b, caracteres=False):
    """Target encoding de vendedor y categoría + probabilidad del texto combinado.
    idx_a: filas de entrenamiento, idx_b: filas a predecir (posiciones en `todo`)."""
    texto = texto_combinado(todo)
    sub = lambda s, idx: s.iloc[idx].reset_index(drop=True)
    return {
        "te_seller_id": target_encoding(sub(todo["seller_id"], idx_a), y[idx_a], sub(todo["seller_id"], idx_b)),
        "te_category_id": target_encoding(sub(todo["category_id"], idx_a), y[idx_a], sub(todo["category_id"], idx_b)),
        "prob_combinado": prob_texto(sub(texto, idx_a), y[idx_a], sub(texto, idx_b), caracteres),
    }


def alinear_categorias(A, B):
    # las categorías de B (validación o test) tienen que ser las mismas que las de A (entrenamiento)
    for c in [c for c in CAT if c in A.columns]:
        cats = pd.Categorical(A[c]).categories
        A[c] = pd.Categorical(A[c], categories=cats)
        B[c] = pd.Categorical(B[c].where(B[c].isin(cats)), categories=cats)
    return A, B


def armar_conjunto(X_todo, piezas, idx_a, idx_b):
    """Features + piezas, listas para el modelo. Devuelve (A, B) = (entrenamiento, a predecir)."""
    A = X_todo.iloc[idx_a].reset_index(drop=True)
    B = X_todo.iloc[idx_b].reset_index(drop=True)
    for n, (a, b) in piezas.items():
        A[n], B[n] = a, b
    return alinear_categorias(A, B)

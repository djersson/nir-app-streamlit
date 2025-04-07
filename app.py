import streamlit as st
import pandas as pd
import numpy as np
import struct
import io
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.spatial import ConvexHull
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import pickle

st.set_page_config(page_title="Análisis NIR - Laboratorio Metalúrgico", layout="wide")
st.title("🔬 Análisis de resultados por Espectroscopía NIR")
st.markdown("""
**Laboratorio Metalúrgico - Minera Chinalco Perú (2025)**  
Desarrollador: Jersson Dávila R.
""")

st.markdown("---")

# === Funciones auxiliares ===
def read_asd_spectrum(file, offset=484):
    content = file.read()
    float_count = (len(content) - offset) // 4
    data = content[offset:offset + float_count * 4]
    reflectance = struct.unpack('<' + 'f' * float_count, data)
    return np.array(reflectance)

def strict_clean(spectrum):
    spectrum = np.array(spectrum)
    mask = np.isfinite(spectrum) & (spectrum >= 0) & (spectrum <= 2)
    if not np.all(mask):
        spectrum[~mask] = np.interp(np.flatnonzero(~mask), np.flatnonzero(mask), spectrum[mask])
    return spectrum

def normalize_convex_hull(wavelengths, spectrum):
    points = np.column_stack((wavelengths, spectrum))
    hull = ConvexHull(points)
    hull_vertices = hull.vertices[np.argsort(wavelengths[hull.vertices])]
    hull_wl = wavelengths[hull_vertices]
    hull_ref = spectrum[hull_vertices]
    hull_interp = np.interp(wavelengths, hull_wl, hull_ref)
    return spectrum / hull_interp

def min_max_scale(spectrum):
    return (spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum))

def auc_difference(a, b):
    return abs(np.trapz(a) - np.trapz(b))

def mean_absolute_error(a, b):
    return np.mean(np.abs(np.array(a) - np.array(b)))

def pearson_corr(a, b):
    return np.corrcoef(a, b)[0, 1]

# === Subida de archivos ===
actualizar = st.sidebar.button("🔁 Actualizar resultados")
st.sidebar.header("Paso 1: Subir archivos .asd")
uploaded_files = st.sidebar.file_uploader("Selecciona uno o varios archivos .asd", type="asd", accept_multiple_files=True)

nombre_patron = None
if uploaded_files:
    nombre_patron = st.sidebar.selectbox("Paso 2: Seleccionar archivo patrón", [f.name for f in uploaded_files])

if uploaded_files and actualizar:
    wl_start, wl_end = 350, 2500
    num_target_points = 2151
    wavelengths = np.linspace(wl_start, wl_end, num_target_points)

    spectra_data = []
    for file in uploaded_files:
        file.seek(0)
        raw = read_asd_spectrum(file)
        raw_clean = strict_clean(raw)
        wl_original = np.linspace(wl_start, wl_end, len(raw_clean))
        interp = np.interp(wavelengths, wl_original, raw_clean)
        smooth = savgol_filter(interp, window_length=15, polyorder=3)
        hull_norm = normalize_convex_hull(wavelengths, smooth)
        scaled = min_max_scale(hull_norm)

        spectra_data.append({
            "nombre": file.name,
            "original": raw_clean,
            "interpolado": interp,
            "suavizado": smooth,
            "norm_hullq": hull_norm,
            "minmax": scaled,
            "num_puntos": len(raw_clean),
            "resolucion": round((wl_end - wl_start) / len(raw_clean), 4),
            "rango": wl_end - wl_start,
            "inicio": wl_start,
            "final": wl_end
        })

    patron = next(s for s in spectra_data if s["nombre"] == nombre_patron)

    # === Cálculo ===
    distancias = []
    similitudes = []
    pearsons = []
    aucs = []
    maes = []
    interpretaciones = []

    for s in spectra_data:
        if s["nombre"] != patron["nombre"]:
            d = np.linalg.norm(patron["suavizado"] - s["suavizado"])
            c = cosine_similarity([patron["suavizado"]], [s["suavizado"]])[0][0]
            p = pearson_corr(patron["suavizado"], s["suavizado"])
            a = auc_difference(patron["suavizado"], s["suavizado"])
            m = mean_absolute_error(patron["suavizado"], s["suavizado"])

            texto = f"Distancia: {d:.2f} | Coseno: {c:.3f} | Pearson: {p:.3f} | AUC: {a:.3f} | MAE: {m:.4f}"
            distancias.append((s["nombre"], d))
            similitudes.append((s["nombre"], c))
            pearsons.append((s["nombre"], p))
            aucs.append((s["nombre"], a))
            maes.append((s["nombre"], m))
            interpretaciones.append((s["nombre"], texto))

    df_export = pd.DataFrame({
        "Archivo": [x[0] for x in distancias],
        "Distancia Euclidiana": [x[1] for x in distancias],
        "Similitud de Coseno": [x[1] for x in similitudes],
        "Correlación Pearson": [x[1] for x in pearsons],
        "Diferencia AUC": [x[1] for x in aucs],
        "Error Absoluto Medio": [x[1] for x in maes],
        "Interpretación": [x[1] for x in interpretaciones]
    })

    st.markdown("### 📏 Distancia Euclidiana respecto al Patrón")
    st.dataframe(df_export[["Archivo", "Distancia Euclidiana"]])

    st.markdown("### 📐 Similitud de Coseno respecto al Patrón")
    st.dataframe(df_export[["Archivo", "Similitud de Coseno"]])

    st.markdown("### 🔁 Correlación de Pearson")
    st.dataframe(df_export[["Archivo", "Correlación Pearson"]])

    st.markdown("### 🧮 Diferencia de Área bajo la Curva (AUC)")
    st.dataframe(df_export[["Archivo", "Diferencia AUC"]])

    st.markdown("### 📉 Error Absoluto Medio")
    st.dataframe(df_export[["Archivo", "Error Absoluto Medio"]])

    st.markdown("### 🧠 Interpretación automática")
    for i in range(len(df_export)):
        archivo = df_export.iloc[i]["Archivo"]
        st.markdown(f"**{archivo}** → {df_export.iloc[i]['Interpretación']}")

    st.markdown("""
---
### ✅ Recomendaciones
<ul>
<li><b>Distancia Euclidiana &gt; 6</b>: Considerar acción correctiva.</li>
<li><b>Similitud de Coseno &lt; 0.5</b>: Indica un cambio significativo en la forma del espectro.</li>
<li><b>Correlación de Pearson &lt; 0.7</b>: Señal de variación significativa en el comportamiento espectral.</li>
<li><b>Diferencia de AUC &gt; 0.1</b>: Puede reflejar cambios en concentración o pureza.</li>
<li><b>Error Absoluto Medio &gt; 0.03</b>: Diferencias distribuidas a lo largo del espectro.</li>
<li><b>Revisar condiciones</b> de muestreo, dilución o contaminación del reactivo.</li>
</ul>

### 🧾 Leyenda para interpretación
<b>Distancia Euclidiana:</b>
<ul>
<li>&lt; 3 : Muy similar al patrón</li>
<li>3–6 : Moderadamente diferente</li>
<li>&gt; 6 : Diferencia significativa</li>
</ul>
<b>Similitud de Coseno:</b>
<ul>
<li>&gt; 0.9 : Forma prácticamente idéntica</li>
<li>0.7–0.9 : Forma parecida</li>
<li>&lt; 0.7 : Forma distinta o alterada</li>
</ul>
<b>Correlación de Pearson:</b>
<ul>
<li>&gt; 0.9 : Muy alta correlación</li>
<li>0.7–0.9 : Correlación moderada</li>
<li>&lt; 0.7 : Baja correlación</li>
</ul>
<b>Diferencia de AUC:</b>
<ul>
<li>&lt; 0.05 : Muy similares en área</li>
<li>0.05–0.1 : Ligeramente diferentes</li>
<li>&gt; 0.1 : Diferencia notable en contenido</li>
</ul>
<b>Error Absoluto Medio:</b>
<ul>
<li>&lt; 0.01 : Diferencia mínima</li>
<li>0.01–0.03 : Diferencia moderada</li>
<li>&gt; 0.03 : Diferencia significativa</li>
</ul>
""", unsafe_allow_html=True)

else:
    st.info("Sube archivos .asd para procesarlos.")

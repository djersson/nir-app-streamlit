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

# === Subida de archivos ===
actualizar = st.sidebar.button("🔁 Actualizar resultados")
st.sidebar.header("Paso 1: Subir archivos .asd")
uploaded_files = st.sidebar.file_uploader("Selecciona uno o varios archivos .asd", type="asd", accept_multiple_files=True)

if uploaded_files and actualizar:
    wl_start, wl_end = 350, 2500
    num_target_points = 2151
    wavelengths = np.linspace(wl_start, wl_end, num_target_points)

    spectra_data = []
    for file in uploaded_files:
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

    nombre_patron = st.sidebar.selectbox("Paso 2: Seleccionar archivo patrón", [s["nombre"] for s in spectra_data])
    patron = next(s for s in spectra_data if s["nombre"] == nombre_patron)

    # === Gráfico ===
    st.subheader("📈 Comparación de espectros normalizados")
    fig, ax = plt.subplots(figsize=(5, 2.3))
    ax.plot(wavelengths, patron["minmax"], label=f"PATRÓN: {patron['nombre']}", linewidth=2)
    for s in spectra_data:
        if s["nombre"] != patron["nombre"]:
            ax.plot(wavelengths, s["minmax"], label=s["nombre"])
    ax.set_xlabel("Longitud de onda (nm)")
    ax.set_ylabel("Reflectancia (0-1)")
    ax.set_title("Espectros NIR normalizados")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # === Tabla resumen de espectros ===
    st.subheader("📋 Tabla resumen de archivos cargados")
    df_resumen = pd.DataFrame([
        {
            "Archivo": s["nombre"],
            "# Puntos espectrales": s["num_puntos"],
            "Long. de onda inicial (nm)": s["inicio"],
            "Long. de onda final (nm)": s["final"],
            "Rango espectral (nm)": s["rango"],
            "Resolución estimada (nm/punto)": s["resolucion"]
        } for s in spectra_data
    ])
    st.dataframe(df_resumen, use_container_width=True)

    # === Cálculo ===

    distancias = []
    similitudes = []
    interpretaciones = []

    for s in spectra_data:
        if s["nombre"] != patron["nombre"]:
            d = np.linalg.norm(patron["suavizado"] - s["suavizado"])
            c = cosine_similarity([patron["suavizado"]], [s["suavizado"]])[0][0]
            texto = f"Distancia: {d:.2f} | Coseno: {c:.3f}"
            distancias.append((s["nombre"], d))
            similitudes.append((s["nombre"], c))
            interpretaciones.append((s["nombre"], texto))

    df_export = pd.DataFrame({
        "Archivo": [x[0] for x in distancias],
        "Distancia Euclidiana": [x[1] for x in distancias],
        "Similitud de Coseno": [x[1] for x in similitudes],
        "Interpretación": [x[1] for x in interpretaciones]
    })

    st.markdown("📏 Distancia Euclidiana respecto al Patrón", unsafe_allow_html=True)
    st.dataframe(df_export[["Archivo", "Distancia Euclidiana"]], use_container_width=True)

    st.markdown("<h4 style='color:#262730;'>📐 Similitud de Coseno respecto al Patrón</h4>", unsafe_allow_html=True)
    st.dataframe(df_export[["Archivo", "Similitud de Coseno"]], use_container_width=True)

    st.markdown("<h4 style='color:#007ACC;'>🧠 Interpretación automática</h4>", unsafe_allow_html=True)
    for i in range(len(df_export)):
        archivo = df_export.iloc[i]["Archivo"]
        dist = df_export.iloc[i]["Distancia Euclidiana"]
        cos = df_export.iloc[i]["Similitud de Coseno"]

        if dist < 3:
            nivel_dist = "✅ Muy similar al patrón"
        elif dist < 6:
            nivel_dist = "🟡 Moderadamente diferente"
        else:
            nivel_dist = "🔴 Muy diferente"

        if cos > 0.9:
            nivel_cos = "✅ Forma prácticamente idéntica"
        elif cos > 0.7:
            nivel_cos = "🟡 Forma parecida"
        else:
            nivel_cos = "🔴 Forma distinta o alterada"

        st.markdown(f"**{archivo}** → Distancia: {dist:.2f} {nivel_dist} | Coseno: {cos:.3f} {nivel_cos}")

    st.markdown("""
---
<h3 style='color:#007ACC;'>✅ Recomendaciones</h3>
<ul>
<li><b>Distancia Euclidiana &gt; 6</b>: Considerar acción correctiva.</li>
<li><b>Similitud de Coseno &lt; 0.5</b>: Indica un cambio significativo en la forma del espectro.</li>
<li><b>Revisar condiciones</b> de muestreo, dilución o contaminación del reactivo.</li>
</ul>

<h3 style='color:#007ACC;'>🧾 Leyenda para interpretación</h3>
<b>Distancia Euclidiana:</b>
<ul>
<li>&lt; 3 : Muy similar al patrón</li>
<li>3-6 : Moderadamente diferente</li>
<li>&gt; 6 : Diferencia significativa</li>
</ul>
<b>Similitud de Coseno:</b>
<ul>
<li>&gt; 0.9 : Forma prácticamente idéntica</li>
<li>0.7-0.9 : Forma parecida</li>
<li>&lt; 0.7 : Forma distinta o alterada</li>
</ul>
""", unsafe_allow_html=True)

else:
    st.info("Sube archivos .asd para procesarlos.")

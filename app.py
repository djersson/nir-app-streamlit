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
actualizar = st.sidebar.button("🤁 Actualizar resultados")
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

    # === Gráfico resumen ===
    st.subheader("📈 Comparación de espectros normalizados")
    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.plot(wavelengths, patron["suavizado"], label=f"PATRÓN: {patron['nombre']}", linewidth=1.5)
    for s in spectra_data:
        if s["nombre"] != patron["nombre"]:
            ax.plot(wavelengths, s["suavizado"], label=s["nombre"])
    ax.set_xlabel("Longitud de onda (nm)", fontsize=8)
    ax.set_ylabel("Reflectancia (suavizada)", fontsize=8)
    ax.set_title("Espectros NIR suavizados", fontsize=9)
    ax.legend(fontsize=6)
    ax.grid(True)
    ax.tick_params(labelsize=6)
    st.pyplot(fig)

    # === Cálculo ===
    resultados = []
    for s in spectra_data:
        if s["nombre"] != patron["nombre"]:
            d = np.linalg.norm(patron["suavizado"] - s["suavizado"])
            c = cosine_similarity([patron["suavizado"]], [s["suavizado"]])[0][0]
            p = pearson_corr(patron["suavizado"], s["suavizado"])
            a = auc_difference(patron["suavizado"], s["suavizado"])
            m = mean_absolute_error(patron["suavizado"], s["suavizado"])

            def color_icono(valor, niveles):
                if niveles[0](valor): return "✅"
                if niveles[1](valor): return "🟡"
                return "🔴"

            icon_dist = color_icono(d, [lambda x: x<3, lambda x: x<6])
            icon_cos = color_icono(c, [lambda x: x>0.9, lambda x: x>0.7])
            icon_pear = color_icono(p, [lambda x: x>0.9, lambda x: x>0.7])
            icon_auc = color_icono(a, [lambda x: x<0.05, lambda x: x<0.1])
            icon_mae = color_icono(m, [lambda x: x<0.01, lambda x: x<0.03])

            # Evaluación global tipo semáforo
            icons = [icon_dist, icon_cos, icon_pear, icon_auc, icon_mae]
            rojo = icons.count("🔴")
            verde_amarillo = icons.count("✅") + icons.count("🟡")
            if verde_amarillo >= 3:
                resumen = "✅ Aproximadamente igual al patrón"
            elif rojo >= 3:
                resumen = "🔴 Totalmente diferente"
            else:
                resumen = "🟡 Moderadamente diferente"

            resultados.append({
                "Archivo": s["nombre"],
                "Distancia Euclidiana": round(d, 4),
                "Similitud de Coseno": round(c, 4),
                "Correlación Pearson": round(p, 4),
                "Diferencia AUC": round(a, 4),
                "Error Absoluto Medio": round(m, 4),
                "Evaluación": resumen
            })

    df_final = pd.DataFrame(resultados)

    st.markdown("### 📊 Resultados numéricos")
    st.dataframe(df_final, use_container_width=True)

    st.markdown("""
---
### ✅ Recomendaciones
- **Distancia Euclidiana > 6**: Considerar acción correctiva.
- **Similitud de Coseno < 0.5**: Cambio significativo en forma espectral.
- **Pearson < 0.5**: Baja correlación lineal.
- **AUC > 0.1**: Diferencia notoria bajo la curva.
- **MAE > 0.03**: Error medio absoluto alto.
- **Verificar** condiciones de muestreo, dilución o contaminación.

---
### 📜 Leyenda para interpretación
**Distancia Euclidiana:**
- ✅ < 3: Muy similar al patrón  
- 🟡 3–6: Moderadamente diferente  
- 🔴 > 6: Diferencia significativa

**Similitud de Coseno / Pearson:**
- ✅ > 0.9: Forma prácticamente idéntica  
- 🟡 0.7–0.9: Forma parecida  
- 🔴 < 0.7: Forma distinta o alterada

**AUC (Diferencia de área bajo la curva):**
- ✅ < 0.05: Prácticamente igual  
- 🟡 0.05–0.1: Leve diferencia  
- 🔴 > 0.1: Diferencia significativa

**MAE (Error Absoluto Medio):**
- ✅ < 0.01: Muy bajo  
- 🟡 0.01–0.03: Tolerable  
- 🔴 > 0.03: Alto
""", unsafe_allow_html=True)

else:
    st.info("Sube archivos .asd para procesarlos.")

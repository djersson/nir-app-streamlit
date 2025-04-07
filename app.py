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

def curve_length(wl, ref):
    return np.sum(np.sqrt(np.diff(wl)**2 + np.diff(ref)**2))

def slope_change_count(ref):
    deriv = np.diff(ref)
    return np.sum(np.diff(np.sign(deriv)) != 0)

def spectral_rugosity(ref):
    return np.std(np.diff(ref))

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
    ax.plot(wavelengths, patron["interpolado"], label=f"PATRÓN: {patron['nombre']}", linewidth=1.5)
    for s in spectra_data:
        if s["nombre"] != patron["nombre"]:
            ax.plot(wavelengths, s["interpolado"], label=s["nombre"])
    ax.set_xlabel("Longitud de onda (nm)", fontsize=8)
    ax.set_ylabel("Reflectancia (interpolada)", fontsize=8)
    ax.set_title("Espectros NIR interpolados", fontsize=9)
    ax.legend(fontsize=6)
    ax.grid(True)
    ax.tick_params(labelsize=6)
    st.pyplot(fig)

    # === Cálculo con ponderación ===
    resultados = []
    for s in spectra_data:
        if s["nombre"] != patron["nombre"]:
            d = np.linalg.norm(patron["interpolado"] - s["interpolado"])
            c = cosine_similarity([patron["interpolado"]], [s["interpolado"]])[0][0]
            p = pearson_corr(patron["interpolado"], s["interpolado"])
            a = auc_difference(patron["interpolado"], s["interpolado"])
            m = mean_absolute_error(patron["interpolado"], s["interpolado"])
            
            # Nuevas métricas de ruta
            cl_ref = curve_length(wavelengths, s["interpolado"])
            cl_pat = curve_length(wavelengths, patron["interpolado"])
            cl_diff = abs(cl_ref - cl_pat)

            sc_ref = slope_change_count(s["interpolado"])
            sc_pat = slope_change_count(patron["interpolado"])
            sc_diff = abs(sc_ref - sc_pat)

            rug_ref = spectral_rugosity(s["interpolado"])
            rug_pat = spectral_rugosity(patron["interpolado"])
            rug_diff = abs(rug_ref - rug_pat)

            # Normalización
            d_norm = min(d / 20, 1)
            c_norm = 1 - max(min(c, 1), 0)
            p_norm = 1 - max(min(p, 1), 0)
            a_norm = min(a / 60, 1)
            m_norm = min(m / 0.1, 1)
            cl_norm = min(cl_diff / 10, 1)
            sc_norm = min(sc_diff / 100, 1)
            rug_norm = min(rug_diff / 0.05, 1)

            # PESOS actualizados incluyendo métricas de ruta
            score = (
                0.15 * d_norm +
                0.02 * c_norm +
                0.03 * p_norm +
                0.25 * a_norm +
                0.05 * m_norm +
                0.20 * cl_norm +
                0.30 * sc_norm +
                0.05 * rug_norm
            )

            if score <= 0.3:
                evaluacion = "✅ Aproximadamente igual al patrón"
            elif score <= 0.6:
                evaluacion = "🟡 Moderadamente diferente"
            else:
                evaluacion = "🔴 Totalmente diferente"

            resultados.append({
                "Archivo": s["nombre"],
                "Distancia Euclidiana": round(d, 4),
                "Similitud de Coseno": round(c, 4),
                "Correlación Pearson": round(p, 4),
                "Diferencia AUC": round(a, 4),
                "Error Absoluto Medio": round(m, 4),
                "Δ Longitud Curva": round(cl_diff, 4),
                "Δ Cambios de Pendiente": sc_diff,
                "Δ Rugosidad": round(rug_diff, 4),
                "Score ponderado": round(score, 4),
                "#📊 % de similitud": round((1 - score) * 100, 2)
            })

    df_final = pd.DataFrame(resultados)

    st.markdown("### 📊 Resultados numéricos")
    st.dataframe(df_final, use_container_width=True)

    st.markdown("""
---
### ✅ Recomendaciones
- Evaluar con mayor detalle las muestras marcadas como 🟡 o 🔴.
- Comparar condiciones de muestreo, dilución, lote y conservación.
- Usar patrón actualizado de referencia si el reactivo ha cambiado de proveedor o formulación.

---
### 📜 Leyenda para interpretación (% de similitud)
- ✅ ≥ 70%: Aproximadamente igual al patrón  
- 🟡 40%–69%: Moderadamente diferente  
- 🔴 < 40%: Totalmente diferente
""", unsafe_allow_html=True)

else:
    st.info("Sube archivos .asd para procesarlos.")

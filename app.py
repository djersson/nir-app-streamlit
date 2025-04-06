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
from fpdf import FPDF
from io import BytesIO
import tempfile

st.set_page_config(page_title="Análisis NIR - Laboratorio Metalúrgico", layout="wide")
st.title("🔬 Análisis de Reactivos por Espectroscopía NIR")
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
st.sidebar.header("Paso 1: Subir archivos .asd")
uploaded_files = st.sidebar.file_uploader("Selecciona uno o varios archivos .asd", type="asd", accept_multiple_files=True)

if uploaded_files:
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
    fig, ax = plt.subplots(figsize=(10, 5))
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

    # === Cálculo ===
    distancias = []
    similitudes = []
    interpretaciones = []

    for s in spectra_data:
        if s["nombre"] != patron["nombre"]:
            d = np.linalg.norm(patron["suavizado"] - s["suavizado"])
            c = cosine_similarity([patron["suavizado"]], [s["suavizado"]])[0][0]
            icono_dist = "✅" if d < 3 else "🟡" if d < 6 else "🔴"
            icono_cos = "✅" if c > 0.9 else "🟡" if c > 0.7 else "🔴"
            texto = f"{icono_dist} Distancia: {d:.2f} | {icono_cos} Coseno: {c:.3f}"
            distancias.append((s["nombre"], d))
            similitudes.append((s["nombre"], c))
            interpretaciones.append((s["nombre"], texto))

    df_export = pd.DataFrame({
        "Archivo": [x[0] for x in distancias],
        "Distancia Euclidiana": [x[1] for x in distancias],
        "Similitud de Coseno": [x[1] for x in similitudes],
        "Interpretación": [x[1] for x in interpretaciones]
    })

    st.subheader("📏 Resultados e interpretación")
    st.dataframe(df_export, use_container_width=True)

    # === PDF ===
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 20)
    pdf.ln(80)
    pdf.cell(0, 15, "Reporte de Espectroscopía NIR", ln=True, align="C")
    pdf.set_font("Arial", size=14)
    pdf.cell(0, 10, "Laboratorio Metalúrgico - Minera Chinalco Perú", ln=True, align="C")
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 10, f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")

    pdf.add_page()
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 10, f"Fecha de análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 10, "Interpretación de Resultados:", ln=True)
    pdf.set_font("Arial", size=10)
    for i in range(len(df_export)):
        row = df_export.iloc[i]
        pdf.multi_cell(0, 8, f"{row['Archivo']}\n→ Distancia: {row['Distancia Euclidiana']:.2f} | Coseno: {row['Similitud de Coseno']:.3f}\n→ {row['Interpretación']}\n")

    pdf.ln(5)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 10, "Recomendaciones:", ln=True)
    pdf.set_font("Arial", size=10)
    recomendaciones = (
        "- Si la distancia euclidiana es mayor a 6, considerar acción correctiva.\n"
        "- Si la similitud de coseno es menor a 0.5, revisar la composición, dilución o pureza del reactivo.\n"
        "- Confirmar condiciones de muestreo y preparación de la muestra antes del análisis."
    )
    pdf.multi_cell(0, 8, recomendaciones)

    output_pdf_bytes = pdf.output(dest='S').encode('latin1')
    st.download_button(
        label="📄 Descargar archivo PDF",
        data=output_pdf_bytes,
        file_name=f"reporte_NIR_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf"
    )
else:
    st.info("Sube archivos .asd para procesarlos.")

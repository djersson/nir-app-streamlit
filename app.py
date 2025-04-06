from fpdf import FPDF
from io import BytesIO
from datetime import datetime
import streamlit as st

# Asegúrate de que 'pdf' esté correctamente definido antes de usarlo
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", "B", 20)
pdf.ln(80)
pdf.cell(0, 15, "Reporte de Espectroscopía NIR", ln=True, align="C")
pdf.set_font("Arial", size=14)
pdf.cell(0, 10, "Laboratorio Metalúrgico - Minera Chinalco Perú", ln=True, align="C")
pdf.set_font("Arial", size=10)
pdf.cell(0, 10, f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")

# === Página de contenido ===
pdf.add_page()
pdf.set_font("Arial", size=10)
pdf.cell(0, 10, f"Fecha de análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
pdf.ln(5)

# Insertar imagen del gráfico (opcional si tienes temp_img.name)
pdf.set_font("Arial", "B", 11)
pdf.cell(0, 10, "Gráfico de Comparación de Espectros:", ln=True)
# pdf.image(temp_img.name, w=180)  # Asegúrate de definir temp_img si lo vas a usar
pdf.ln(5)

pdf.set_font("Arial", "B", 11)
pdf.cell(0, 10, "Interpretación de Resultados:", ln=True)
pdf.set_font("Arial", size=10)

# Simulación de resultados si no tienes df_export aún
# for i in range(len(df_export)):
#     row = df_export.iloc[i]
#     pdf.multi_cell(0, 8, f"{row['Archivo']}\n→ Distancia: {row['Distancia Euclidiana']:.2f} | Coseno: {row['Similitud de Coseno']:.3f}\n→ {row['Interpretación']}\n", border=0)

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

# === Exportar como bytes correctamente ===
output_pdf_bytes = pdf.output(dest='S').encode('latin1')

st.download_button(
    label="📄 Descargar archivo PDF",
    data=output_pdf_bytes,
    file_name=f"reporte_NIR_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
    mime="application/pdf"
)

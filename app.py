
# app.py (fragmento corregido)
from fpdf import FPDF
from io import BytesIO
from datetime import datetime
import os
import tempfile

# Simulación del gráfico exportado
class PDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Página {self.page_no()}", align="C")

pdf = PDF()
pdf.add_page()
pdf.set_font("Arial", "B", 20)
pdf.ln(80)
pdf.cell(0, 15, "Reporte de Espectroscopía NIR", ln=True, align="C")
pdf.set_font("Arial", size=14)
pdf.cell(0, 10, "Laboratorio Metalúrgico - Minera Chinalco Perú", ln=True, align="C")
pdf.set_font("Arial", size=10)
pdf.cell(0, 10, f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")

# Página de contenido
pdf.add_page()
pdf.set_font("Arial", size=10)
pdf.cell(0, 10, f"Fecha de análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
pdf.ln(5)

pdf.set_font("Arial", "B", 11)
pdf.cell(0, 10, "Gráfico de Comparación de Espectros:", ln=True)
# pdf.image(temp_img.name, w=180)
pdf.ln(5)

pdf.set_font("Arial", "B", 11)
pdf.cell(0, 10, "Interpretación de Resultados:", ln=True)
pdf.set_font("Arial", size=10)
# Ejemplo de interpretación ficticia
pdf.multi_cell(0, 8, "Reactivo H75(50%)
→ Distancia: 5.31 | Coseno: 0.030
→ 🔴 Muy diferente | 🔴 Forma distinta o alterada
")

pdf.ln(5)
pdf.set_font("Arial", "B", 11)
pdf.cell(0, 10, "Recomendaciones:", ln=True)
pdf.set_font("Arial", size=10)
recomendaciones = (
    "- Si la distancia euclidiana es mayor a 6, considerar acción correctiva.
"
    "- Si la similitud de coseno es menor a 0.5, revisar la composición, dilución o pureza del reactivo.
"
    "- Confirmar condiciones de muestreo y preparación de la muestra antes del análisis."
)
pdf.multi_cell(0, 8, recomendaciones)

output_pdf = BytesIO()
pdf.output(output_pdf)
output_pdf.seek(0)

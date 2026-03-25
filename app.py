import pandas as pd
import ollama
import streamlit as st

# Cargar dataset
df = pd.read_csv("Pénsum - Ingeniería de sistemas y computación.csv", sep=";")
df.columns = df.columns.str.strip()

# Convertir filas en texto contextual
def convertir_a_texto(row):

    # Detectar automáticamente columnas de requisitos
    columnas_requisitos = [col for col in df.columns if "Requisito" in col]

    requisitos = []

    for col in columnas_requisitos:
        if pd.notna(row[col]):
            requisitos.append(str(row[col]))

    requisitos_texto = ", ".join(requisitos) if requisitos else "Sin prerrequisitos"

    return f"""
    Materia: {row['Nombre']}
    Código: {row['Codigo']}
    Semestre: {row['Periodo']}
    Créditos: {row['Creditos']}
    Prerrequisitos: {requisitos_texto}
    """
documentos = df.apply(convertir_a_texto, axis=1).tolist()

# Función para generar respuesta
def generar_respuesta(pregunta):

    contexto = "\n".join(documentos) 

# Cambiar la carrera para pruebas
    prompt = f"""
    Actúa como un asesor académico experto en Ingeniería de sistemas y computación.

    Tu función es:
    - Recomendar materias, teniendo en cuenta los requisitos
    - Decir si un estudiante puede inscribir una materia
    - Orientar matricula según semestre (Minimo 10 creditos y maximo 18 creditos por semestre)

    Pensum oficial:
    {contexto}

    Pregunta del estudiante:
    {pregunta}

    Responde de forma clara, concisa, profesional y académica.
    """

    response = ollama.generate(
        model="llama3.1:8b",
        prompt=prompt
    )

    return response["response"]

# Interfaz
st.title("UniguIA")
st.write("¡Mejora tu proceso de matricula academica en la UdeC con IA!")

pregunta_usuario = st.text_input("¿Cúal es tu pregunta?")

if st.button("Enviar"):
    if pregunta_usuario:
        respuesta = generar_respuesta(pregunta_usuario)
        st.write("### Respuesta:")
        st.write(respuesta)
    else:
        st.warning("Por favor escribe una pregunta.")
import re
import pandas as pd
import ollama
from rapidfuzz import fuzz
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Configuración de carreras y prompts personalizados

CARRERAS = {
    "Administración de empresas": {
        "csv": "Pénsum - Administración de empresas.csv",
        "prompt": (
            "Actúa como un asesor académico experto en Administración de Empresas. "
            "Tu función es:\n"
            "- Listar y recomendar materias teniendo en cuenta los requisitos\n"
            "- Decir si un estudiante puede inscribir una materia\n"
            "- Orientar la matrícula según semestre (mínimo 10 créditos y máximo 18 créditos por semestre)"
        ),
    },
    "Contaduría pública": {
        "csv": "Pénsum - Contaduría pública.csv",
        "prompt": (
            "Actúa como un asesor académico experto en Contaduría Pública. "
            "Tu función es:\n"
            "- Listar y recomendar materias teniendo en cuenta los requisitos\n"
            "- Decir si un estudiante puede inscribir una materia\n"
            "- Orientar la matrícula según semestre (mínimo 10 créditos y máximo 18 créditos por semestre)"
        ),
    },
    "Ingeniería de sistemas y computación": {
        "csv": "Pénsum - Ingeniería de sistemas y computación.csv",
        "prompt": (
            "Actúa como un asesor académico experto en Ingeniería de Sistemas y Computación. "
            "Tu función es:\n"
            "- Listar y recomendar materias teniendo en cuenta los requisitos\n"
            "- Decir si un estudiante puede inscribir una materia\n"
            "- Orientar la matrícula según semestre (mínimo 10 créditos y máximo 18 créditos por semestre)"
        ),
    },
    "Ingeniería industrial": {
        "csv": "Pénsum - Ingeniería industrial.csv",
        "prompt": (
            "Actúa como un asesor académico experto en Ingeniería Industrial. "
            "Tu función es:\n"
            "- Listar y recomendar materias teniendo en cuenta los requisitos\n"
            "- Decir si un estudiante puede inscribir una materia\n"
            "- Orientar la matrícula según semestre (mínimo 10 créditos y máximo 18 créditos por semestre)"
        ),
    },
    "Ingeniería mecatrónica": {
        "csv": "Pénsum - Ingeniería mecatrónica.csv",
        "prompt": (
            "Actúa como un asesor académico experto en Ingeniería Mecatrónica. "
            "Tu función es:\n"
            "- Listar y recomendar materias teniendo en cuenta los requisitos\n"
            "- Decir si un estudiante puede inscribir una materia\n"
            "- Orientar la matrícula según semestre (mínimo 10 créditos y máximo 18 créditos por semestre)"
        ),
    },
}

MAX_MATERIAS_CONTEXTO = 30
UMBRAL_SIMILITUD = 55

# Detección por similitud en palabras (Semestres)

SEMESTRES_TEXTO = {
    "primer":   1, "primero":  1, "1er": 1, "1ro": 1,
    "segundo":  2, "2do": 2,
    "tercer":   3, "tercero":  3, "3er": 3, "3ro": 3,
    "cuarto":   4, "4to": 4,
    "quinto":   5, "5to": 5,
    "sexto":    6, "6to": 6,
    "séptimo":  7, "septimo": 7, "7mo": 7,
    "octavo":   8, "8vo": 8,
    "noveno":   9, "9no": 9,
    "décimo":  10, "decimo": 10, "10mo": 10,
}

# Funciones lógicas del modelo

def detectar_semestre(pregunta: str) -> int | None:
    texto = pregunta.lower()

    # Buscar número directamente: "semestre 3", "3 semestre", "3er semestre"
    match = re.search(r'\b(\d{1,2})\b', texto)
    if match:
        numero = int(match.group(1))
        if 1 <= numero <= 12:
            return numero

    # Buscar palabras ordinales
    for palabra, numero in SEMESTRES_TEXTO.items():
        if palabra in texto:
            return numero

    return None


def cargar_pensum(csv_path: str) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(csv_path, sep=";")
    df.columns = df.columns.str.strip()
    columnas_requisitos = [col for col in df.columns if "Requisito" in col]

    def fila_a_texto(row):
        requisitos = [str(row[col]) for col in columnas_requisitos if pd.notna(row[col])]
        requisitos_texto = ", ".join(requisitos) if requisitos else "Sin prerrequisitos"
        return (
            f"Materia: {row['Nombre']} | "
            f"Código: {row['Codigo']} | "
            f"Semestre: {row['Semestre']} | "
            f"Créditos: {row['Creditos']} | "
            f"Prerrequisitos: {requisitos_texto}"
        )

    documentos = df.apply(fila_a_texto, axis=1).tolist()
    return df, documentos


def filtrar_contexto(
    pregunta: str,
    df: pd.DataFrame,
    documentos: list[str],
    max_items: int = MAX_MATERIAS_CONTEXTO,
) -> tuple[list[str], str | None]:

    semestre = detectar_semestre(pregunta)

    # Filtro exacto por semestre
    if semestre is not None:
        mascara = df["Semestre"].astype(str).str.strip() == str(semestre)
        indices = df[mascara].index.tolist()

        if indices:
            return [documentos[i] for i in indices], str(semestre)

    # Búsqueda fuzzy
    pregunta_lower = pregunta.lower()
    palabras = [p for p in pregunta_lower.split() if len(p) > 3]
    columnas_requisitos = [col for col in df.columns if "Requisito" in col]

    scores = []
    for i, row in df.iterrows():
        nombre = str(row.get("Nombre", "")).lower()
        doc = documentos[i].lower()

        score_nombre = fuzz.partial_ratio(pregunta_lower, nombre)
        score_palabras_nombre = max(
            (fuzz.partial_ratio(p, nombre) for p in palabras), default=0
        )

        requisitos = [
            str(row[col]).lower()
            for col in columnas_requisitos
            if pd.notna(row[col])
        ]
        score_requisitos = max(
            (fuzz.partial_ratio(pregunta_lower, req) for req in requisitos), default=0
        )
        score_palabras_req = max(
            (fuzz.partial_ratio(p, req) for p in palabras for req in requisitos),
            default=0,
        ) if requisitos else 0

        score_keywords = sum(1 for p in palabras if p in doc) * 10

        score_total = (
            score_nombre * 1.5
            + score_palabras_nombre * 1.5
            + score_requisitos * 1.2
            + score_palabras_req * 1.2
            + score_keywords
        )
        scores.append((score_total, documentos[i]))

    scores.sort(key=lambda x: x[0], reverse=True)
    relevantes = [doc for score, doc in scores if score >= UMBRAL_SIMILITUD]

    if len(relevantes) < max_items:
        ya_incluidos = set(relevantes)
        for _, doc in scores:
            if doc not in ya_incluidos:
                relevantes.append(doc)
            if len(relevantes) >= max_items:
                break

    return relevantes[:max_items], None


def generar_respuesta(
    pregunta: str,
    df: pd.DataFrame,
    documentos: list[str],
    prompt_base: str,
) -> str:
    docs_filtrados, semestre_detectado = filtrar_contexto(pregunta, df, documentos)
    contexto = "\n".join(docs_filtrados)

    # Instrucción extra si se detectó semestre
    nota_semestre = (
        f"\nNOTA: La pregunta es sobre el semestre {semestre_detectado}. "
        f"El contexto contiene TODAS las materias de ese semestre. Listarlas completas.\n"
        if semestre_detectado else ""
    )

    prompt = (
        f"{prompt_base}\n"
        f"{nota_semestre}\n"
        f"Pensum (materias relevantes para esta pregunta):\n{contexto}\n\n"
        f"Pregunta del estudiante:\n{pregunta}\n\n"
        "IMPORTANTE: Responde ÚNICAMENTE con información del pensum proporcionado. "
        "Si la materia no aparece en el pensum, dilo explícitamente. "
        "No inventes prerrequisitos ni datos que no estén en el contexto.\n\n"
        "Responde de forma clara, concisa, profesional y académica."
    )

    # Llamar el modelo y configuracion de parametros de tokenización y creatividad
    response = ollama.generate(
        model="llama3.1:8b",
        prompt=prompt,
        options={
            "num_ctx": 4096,
            "num_predict": 512,
            "temperature": 0.3,
        },
    )
    return response["response"]


# API FastAPI

app = FastAPI(title="UniguIA API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Acepta peticiones de cualquier origen
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConsultaRequest(BaseModel):
    carrera: str
    pregunta: str


class ConsultaResponse(BaseModel):
    respuesta: str


@app.get("/carreras")
def listar_carreras() -> list[str]:

    # Devuelve la lista de carreras
    return list(CARRERAS.keys())


@app.post("/consultar", response_model=ConsultaResponse)
def consultar(body: ConsultaRequest) -> ConsultaResponse:

    # Recibe la carrera y la pregunta del estudiante y retorna la respuesta generada por el modelo.
    if body.carrera not in CARRERAS:
        raise HTTPException(
            status_code=400,
            detail=f"Carrera '{body.carrera}' no encontrada. "
                   f"Carreras válidas: {list(CARRERAS.keys())}",
        )

    config = CARRERAS[body.carrera]
    try:
        df, documentos = cargar_pensum(config["csv"])
        respuesta = generar_respuesta(body.pregunta, df, documentos, config["prompt"])
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail=f"No se encontró el archivo CSV para '{body.carrera}'.",
        )
    except ollama.ResponseError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Error al consultar el modelo Ollama: {e}",
        )

    return ConsultaResponse(respuesta=respuesta)
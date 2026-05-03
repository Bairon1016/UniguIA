import re
import unicodedata
import pandas as pd
import ollama
from rapidfuzz import fuzz, process
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Umbral de máximo de materias y parametros de Ollama

MAX_MATERIAS_CONTEXTO = 60
TEMPERATURE = 0.0
NUM_CTX = 8192
NUM_PREDICT = 1028

# Prompts específicos por carrera

CARRERAS = {
    "Administración de empresas": {
        "csv": "Pénsum - Administración de empresas.csv",
        "prompt": (
            "Actúa como un asesor académico experto en Administración de Empresas.\n"
            "Debes analizar el pensum con criterio académico, claridad y orden.\n"
            "Reglas:\n"
            "- Analiza todas las materias que el estudiante mencione.\n"
            "- Relaciona prerrequisitos solo si están explícitos en el contexto.\n"
            "- Si un dato no aparece en el pensum, responde: 'No se especifica en el pensum'.\n"
            "- No inventes prerrequisitos ni supongas información faltante.\n"
            "- No hables sobre notas, parciales, quizes, tareas y derivados. \n"
            "- Si la pregunta incluye varias materias, responde una sección por cada materia.\n"
            "- Mínimo 10 créditos y máximo 18 créditos por semestre.\n"
            "- Solamente si el estudiante lo menciona orienta la matrícula según semestre.\n"
            "- Si la pregunta no tiene respuesta en el pénsum (horarios, costos, profesores, notas, parciales u otros temas), responde ÚNICAMENTE:'Eso está fuera de mi alcance, solo puedo asesorarte sobre materias, prerrequisitos, créditos y semestres del pénsum.'No intentes responder ni parcialmente.\n"
        ),
    },
    "Contaduría pública": {
        "csv": "Pénsum - Contaduría pública.csv",
        "prompt": (
            "Actúa como un asesor académico experto en Contaduría Pública.\n"
            "Debes analizar el pensum con criterio académico, claridad y orden.\n"
            "Reglas:\n"
            "- Analiza todas las materias que el estudiante mencione.\n"
            "- Relaciona prerrequisitos solo si están explícitos en el contexto.\n"
            "- Si un dato no aparece en el pensum, responde: 'No se especifica en el pensum'.\n"
            "- No inventes prerrequisitos ni supongas información faltante.\n"
            "- No hables sobre notas, parciales, quizes, tareas y derivados. \n"
            "- Si la pregunta incluye varias materias, responde una sección por cada materia.\n"
            "- Mínimo 10 créditos y máximo 18 créditos por semestre.\n"
            "- Solamente si el estudiante lo menciona orienta la matrícula según semestre.\n"
            "- Si la pregunta no tiene respuesta en el pénsum (horarios, costos, profesores, notas, parciales u otros temas), responde ÚNICAMENTE:'Eso está fuera de mi alcance, solo puedo asesorarte sobre materias, prerrequisitos, créditos y semestres del pénsum.'No intentes responder ni parcialmente.\n"
        ),
    },
    "Ingeniería de sistemas y computación": {
        "csv": "Pénsum - Ingeniería de sistemas y computación.csv",
        "prompt": (
            "Actúa como un asesor académico experto en Ingeniería de Sistemas y Computación.\n"
            "Debes analizar el pensum con criterio académico, claridad y orden.\n"
            "Reglas:\n"
            "- Siempre da una respuesta completa pero concisa\n"
            "- Analiza todas las materias que el estudiante mencione.\n"
            "- Relaciona prerrequisitos solo si están explícitos en el contexto.\n"
            "- No inventes prerrequisitos ni supongas información faltante.\n"
            "- No hables sobre notas, parciales, quizes, tareas y derivados. \n"
            "- Si la pregunta incluye varias materias, responde una sección concisa por cada materia.\n"
            "- Mínimo 10 créditos y máximo 18 créditos por semestre.\n"
            "- Solamente si el estudiante lo menciona orienta la matrícula según semestre.\n"
            "- Si la pregunta no tiene respuesta en el pénsum (horarios, costos, profesores, notas, parciales u otros temas), responde ÚNICAMENTE:'Eso está fuera de mi alcance, solo puedo asesorarte sobre materias, prerrequisitos, créditos y semestres del pénsum.'No intentes responder ni parcialmente.\n"
            "- Recuerda que la carrera tiene solo 9 semestres, si la pregunta se pasa del semestre 9, responde ÚNICAMENTE:'Este semestre no esta dentro de la carrera.' No intentes responder ni parcialmente.\n"
        ),
    },
    "Ingeniería industrial": {
        "csv": "Pénsum - Ingeniería industrial.csv",
        "prompt": (
            "Actúa como un asesor académico experto en Ingeniería Industrial.\n"
            "Debes analizar el pensum con criterio académico, claridad y orden.\n"
            "Reglas:\n"
            "- Analiza todas las materias que el estudiante mencione.\n"
            "- Relaciona prerrequisitos solo si están explícitos en el contexto.\n"
            "- Si un dato no aparece en el pensum, responde: 'No se especifica en el pensum'.\n"
            "- No inventes prerrequisitos ni supongas información faltante.\n"
            "- No hables sobre notas, parciales, quizes, tareas y derivados. \n"            
            "- Si la pregunta incluye varias materias, responde una sección por cada materia.\n"
            "- Mínimo 10 créditos y máximo 18 créditos por semestre.\n"
            "- Solamente si el estudiante lo menciona orienta la matrícula según semestre.\n"
            "- Si la pregunta no tiene respuesta en el pénsum (horarios, costos, profesores, notas, parciales u otros temas), responde ÚNICAMENTE:'Eso está fuera de mi alcance, solo puedo asesorarte sobre materias, prerrequisitos, créditos y semestres del pénsum.'No intentes responder ni parcialmente.\n"
        ),
    },
    "Ingeniería mecatrónica": {
        "csv": "Pénsum - Ingeniería mecatrónica.csv",
        "prompt": (
            "Actúa como un asesor académico experto en Ingeniería Mecatrónica.\n"
            "Debes analizar el pensum con criterio académico, claridad y orden.\n"
            "Reglas:\n"
            "- Analiza todas las materias que el estudiante mencione.\n"
            "- Relaciona prerrequisitos solo si están explícitos en el contexto.\n"
            "- Si un dato no aparece en el pensum, responde: 'No se especifica en el pensum'.\n"
            "- No inventes prerrequisitos ni supongas información faltante.\n"
            "- No hables sobre notas, parciales, quizes, tareas y derivados. \n"            
            "- Si la pregunta incluye varias materias, responde una sección por cada materia.\n"
            "- Mínimo 10 créditos y máximo 18 créditos por semestre.\n"
            "- Solamente si el estudiante lo menciona orienta la matrícula según semestre.\n"
            "- Si la pregunta no tiene respuesta en el pénsum (horarios, costos, profesores, notas, parciales u otros temas), responde ÚNICAMENTE:'Eso está fuera de mi alcance, solo puedo asesorarte sobre materias, prerrequisitos, créditos y semestres del pénsum.'No intentes responder ni parcialmente.\n"
        ),
    },
}

SEMESTRES_TEXTO = {
    "primer": 1, "primero": 1, "1er": 1, "1ro": 1,
    "segundo": 2, "2do": 2,
    "tercer": 3, "tercero": 3, "3er": 3, "3ro": 3,
    "cuarto": 4, "4to": 4,
    "quinto": 5, "5to": 5,
    "sexto": 6, "6to": 6,
    "septimo": 7, "séptimo": 7, "7mo": 7,
    "octavo": 8, "8vo": 8,
    "noveno": 9, "9no": 9,
}

# Utilidades de texto

def normalize_text(text) -> str:
    """Minúsculas, sin acentos, sin espacios repetidos."""
    if text is None:
        return ""
    if isinstance(text, float) and pd.isna(text):
        return ""
    s = str(text).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s)
    return s

def unique_preserve_order(items):
    return list(dict.fromkeys(items))

def contains_phrase(text: str, phrase: str) -> bool:
    """Busca una frase completa sin que 'i' coincida con 'ii', etc."""
    if not phrase:
        return False
    pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
    return re.search(pattern, text) is not None

# Detección de semestre

def detectar_semestre(pregunta: str) -> int | None:
    texto = normalize_text(pregunta)

    patrones = [
        r"\b(?:semestre|sem)\s*(\d{1,2})\b",
        r"\b(\d{1,2})\s*(?:semestre|sem)\b",
        r"\b(?:primer|primero|1er|1ro)\s+semestre\b",
        r"\b(?:segundo|2do)\s+semestre\b",
        r"\b(?:tercer|tercero|3er|3ro)\s+semestre\b",
        r"\b(?:cuarto|4to)\s+semestre\b",
        r"\b(?:quinto|5to)\s+semestre\b",
        r"\b(?:sexto|6to)\s+semestre\b",
        r"\b(?:septimo|séptimo|7mo)\s+semestre\b",
        r"\b(?:octavo|8vo)\s+semestre\b",
        r"\b(?:noveno|9no)\s+semestre\b",
    ]

    for patron in patrones:
        m = re.search(patron, texto)
        if not m:
            continue
        if m.lastindex:
            numero = int(m.group(1))
            if 1 <= numero <= 99:
                return numero
        else:
            for palabra, numero in SEMESTRES_TEXTO.items():
                if palabra in texto:
                    return numero

    for palabra, numero in SEMESTRES_TEXTO.items():
        if palabra in texto:
            return numero

    return None

# Carga y preparación del pensum

def cargar_pensum(csv_path: str) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(csv_path, sep=";").copy()
    df.columns = df.columns.str.strip()
    df = df.reset_index(drop=True)

    columnas_necesarias = {"Nombre", "Codigo", "Semestre", "Creditos"}
    faltantes = columnas_necesarias - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas obligatorias en {csv_path}: {sorted(faltantes)}")

    columnas_requisitos = [col for col in df.columns if "Requisito" in col]

    df["Nombre_norm"] = df["Nombre"].apply(normalize_text)
    df["Codigo_norm"] = df["Codigo"].apply(normalize_text)
    df["Semestre_norm"] = df["Semestre"].astype(str).apply(normalize_text)
    df["Creditos_norm"] = df["Creditos"].astype(str).apply(normalize_text)

    def construir_requisitos_norm(row):
        reqs = []
        for col in columnas_requisitos:
            if pd.notna(row[col]) and str(row[col]).strip():
                reqs.append(normalize_text(row[col]))
        return " | ".join(reqs)

    df["Requisitos_norm"] = df.apply(construir_requisitos_norm, axis=1)

    def fila_a_texto(row):
        requisitos = [str(row[col]).strip() for col in columnas_requisitos if pd.notna(row[col]) and str(row[col]).strip()]
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

# Extracción de materias mencionadas

def extraer_materias_mencionadas(pregunta: str, df: pd.DataFrame) -> list[int]:
    q = normalize_text(pregunta)
    nombres = df["Nombre_norm"].tolist()
    encontrados = []

    # Coincidencia exacta por frase completa
    for i, nombre in enumerate(nombres):
        if nombre and contains_phrase(q, nombre):
            encontrados.append(i)

    if encontrados:
        return unique_preserve_order(encontrados)

    # Fuzzy fallback si no hay coincidencias exactas
    candidatos = process.extract(
        q,
        nombres,
        scorer=fuzz.token_set_ratio,
        limit=min(10, len(nombres)),
    )

    for _, score, idx in candidatos:
        if score >= 80:
            encontrados.append(idx)

    return unique_preserve_order(encontrados)

# Ranking de contexto

def puntuar_fila(pregunta_norm: str, tokens: list[str], row: pd.Series) -> float:
    nombre = row.get("Nombre_norm", "")
    codigo = row.get("Codigo_norm", "")
    requisitos = row.get("Requisitos_norm", "")
    semestre = row.get("Semestre_norm", "")
    doc_text = f"{nombre} {codigo} {semestre} {requisitos}"

    score_nombre = max(
        fuzz.partial_ratio(pregunta_norm, nombre),
        fuzz.token_set_ratio(pregunta_norm, nombre),
    )

    score_codigo = 0
    if codigo and codigo in pregunta_norm:
        score_codigo = 100

    score_requisitos = 0
    if requisitos:
        score_requisitos = max(
            fuzz.partial_ratio(pregunta_norm, requisitos),
            fuzz.token_set_ratio(pregunta_norm, requisitos),
        )

    score_tokens = sum(1 for t in tokens if t and t in doc_text) * 8

    # Pequeño refuerzo si la pregunta menciona "prerrequisitos", "requisitos", etc.
    score_intencion = 0
    if "prerrequisito" in pregunta_norm or "requisito" in pregunta_norm:
        score_intencion += 10
    if "inscribir" in pregunta_norm or "matricular" in pregunta_norm:
        score_intencion += 6
    if "semestre" in pregunta_norm:
        score_intencion += 4

    return (
        score_nombre * 1.6
        + score_requisitos * 1.4
        + score_codigo * 1.2
        + score_tokens
        + score_intencion
    )

def filtrar_contexto(
    pregunta: str,
    df: pd.DataFrame,
    documentos: list[str],
    max_items: int = MAX_MATERIAS_CONTEXTO,
) -> tuple[list[str], str | None, str]:
    pregunta_norm = normalize_text(pregunta)
    tokens = [t for t in pregunta_norm.split() if len(t) > 2]

    materias_detectadas = extraer_materias_mencionadas(pregunta, df)
    semestre = detectar_semestre(pregunta)

    indices_semestre = []
    if semestre is not None:
        mascara = df["Semestre"].astype(str).str.strip() == str(semestre)
        indices_semestre = df.index[mascara].tolist()

    scores = []
    for idx, row in df.iterrows():
        score = puntuar_fila(pregunta_norm, tokens, row)
        scores.append((idx, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    indices_ranked = [idx for idx, _ in scores]

    # Caso semestre: si no hay materias explícitas, devolver solo ese semestre
    if semestre is not None and not materias_detectadas:
        contexto = [documentos[i] for i in indices_semestre[:max_items]]
        return contexto, str(semestre), "semestre"

    # Caso materia: priorizar materias detectadas + prerrequisitos directos
    if materias_detectadas:
        indices_principales = materias_detectadas.copy()

        indices_relacionados = []
        for idx in materias_detectadas:
            requisitos = df.iloc[idx].get("Requisitos_norm", "")
            for j, row in df.iterrows():
                if row["Nombre_norm"] in requisitos:
                    indices_relacionados.append(j)

        # Poco respaldo adicional, para no meter ruido
        top_extra = [
            idx for idx in indices_ranked
            if idx not in indices_principales and idx not in indices_relacionados
        ]

        indices_finales = unique_preserve_order(
            indices_principales + indices_relacionados + top_extra
        )

        contexto = [documentos[i] for i in indices_finales[:min(max_items, 15)]]
        return contexto, None, "materia"

    # Fallback general: si no detecta nada, usar ranking reducido
    contexto = [documentos[i] for i in indices_ranked[:10]]
    return contexto, None, "general"

# Generación de respuesta

def generar_respuesta(
    pregunta: str,
    df: pd.DataFrame,
    documentos: list[str],
    prompt_base: str,
) -> str:
    docs_filtrados, semestre_detectado, modo = filtrar_contexto(pregunta, df, documentos)

    contexto = "\n".join(docs_filtrados)
    
    # Semestre detectado pero no existe en el pénsum → respuesta directa, sin LLM
    if modo == "semestre" and not docs_filtrados:
        semestres_validos = sorted(df["Semestre"].astype(int).unique())
        return (
            f"El semestre {semestre_detectado} no existe en el pénsum de esta carrera. "
            f"La carrera tiene {len(semestres_validos)} semestres "
            f"(del {semestres_validos[0]} al {semestres_validos[-1]})."
        )

    if modo == "semestre" and semestre_detectado:
        nota = (
            f"NOTA: La consulta es exclusivamente sobre el semestre {semestre_detectado}. "
            f"Responde únicamente con materias de ese semestre. "
            f"Si la pregunta no tiene respuesta en el pénsum, dilo explícitamente y no inventes absolutamente nada."
            f"No menciones materias de otros semestres.\n"
        )
    elif modo == "materia":
        nota = (
            "NOTA: La consulta es sobre materias específicas. "
            "Responde únicamente sobre esas materias y sus prerrequisitos directos. "
            "Si la pregunta no tiene respuesta en el pénsum, dilo explícitamente y no inventes absolutamente nada."
            "No incluyas materias no relacionadas.\n"
        )
    else:
        nota = ""

    prompt = (
        f"{prompt_base}\n"
        f"{nota}\n"
        f"Pensum disponible para análisis:\n{contexto}\n\n"
        f"Pregunta del estudiante:\n{pregunta}\n\n"
        "Instrucciones de respuesta:\n"
        "- Responde con orden y claridad.\n"
        "- No menciones materias que no estén en el contexto.\n"
        "- Si un prerrequisito no está explícito en el contexto, escribe: 'No se especifica en el pensum'.\n"
        "- No inventes datos ni completes con suposiciones.\n"
        "- Prioriza la información del pensum y explica solo con base en ella.\n"
        "- Si la pregunta no tiene respuesta en el pénsum, dilo explícitamente y no inventes absolutamente nada.\n"
    )

    response = ollama.generate(
        model="llama3.1:8b",
        prompt=prompt,
        options={
            "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT,
            "temperature": TEMPERATURE,
        },
    )

    return response["response"]

# API FastAPI

app = FastAPI(title="UniguIA API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    return list(CARRERAS.keys())

@app.post("/consultar", response_model=ConsultaResponse)
def consultar(body: ConsultaRequest) -> ConsultaResponse:
    if body.carrera not in CARRERAS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Carrera '{body.carrera}' no encontrada. "
                f"Carreras válidas: {list(CARRERAS.keys())}"
            ),
        )

    config = CARRERAS[body.carrera]

    try:
        df, documentos = cargar_pensum(config["csv"])
        respuesta = generar_respuesta(body.pregunta, df, documentos, config["prompt"])
        return ConsultaResponse(respuesta=respuesta)

    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail=f"No se encontró el archivo CSV para '{body.carrera}'.",
        )
    except ValueError as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )
    except ollama.ResponseError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Error al consultar el modelo Ollama: {e}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error inesperado: {e}",
        )
    
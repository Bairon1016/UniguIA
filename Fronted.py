import requests
import streamlit as st

# Configuración peticiones al Backend

BACKEND_URL = "http://localhost:8000"


def obtener_carreras() -> list[str]:
    
    # Llama al backend para obtener la lista de carreras
    try:
        response = requests.get(f"{BACKEND_URL}/carreras", timeout=0.001) # Tiempo de respuesta en los GET
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("No se puede conectar al backend. ¿Está corriendo en localhost:8000?")
        return []
    except requests.exceptions.RequestException as e:
        st.error(f"Error al obtener carreras: {e}")
        return []


def consultar_backend(carrera: str, pregunta: str) -> str | None:
    
    # Envía la consulta al backend y retorna la respuesta
    try:
        response = requests.post(
            f"{BACKEND_URL}/consultar",
            json={"carrera": carrera, "pregunta": pregunta},
            timeout=60,  # Tiempo de respuesta de Ollama (Puede tardar depende de los recursos de hardware)
        )
        response.raise_for_status()
        return response.json()["respuesta"]
    except requests.exceptions.ConnectionError:
        st.error("No se puede conectar al backend. ¿Está corriendo en localhost:8000?")
    except requests.exceptions.Timeout:
        st.error("El backend tardó demasiado en responder. Intenta de nuevo.")
    except requests.exceptions.HTTPError as e:
        detalle = e.response.json().get("detail", str(e))
        st.error(f"Error del backend: {detalle}")
    return None


# Interfaz

st.title("UniguIA")
st.write("¡Mejora tu proceso de matrícula académica en la UdeC con IA!")

# Cargar carreras desde el backend
carreras = obtener_carreras()

if not carreras:
    st.stop()

st.subheader("1. Selecciona tu carrera")

if "carrera_seleccionada" not in st.session_state:
    st.session_state.carrera_seleccionada = None

cols = st.columns(len(carreras))
for col, nombre_carrera in zip(cols, carreras):
    with col:
        seleccionada = st.session_state.carrera_seleccionada == nombre_carrera
        if st.button(
            nombre_carrera,
            key=f"btn_{nombre_carrera}",
            type="primary" if seleccionada else "secondary",
            use_container_width=True,
        ):
            st.session_state.carrera_seleccionada = nombre_carrera
            st.rerun()

if st.session_state.carrera_seleccionada:
    st.success(f"Carrera seleccionada: **{st.session_state.carrera_seleccionada}**")

    st.subheader("2. Haz tu pregunta")
    pregunta_usuario = st.text_input("¿Cuál es tu pregunta?")

    if st.button("Enviar", type="primary"):
        if pregunta_usuario:
            with st.spinner("Consultando el modelo..."):
                respuesta = consultar_backend(
                    st.session_state.carrera_seleccionada,
                    pregunta_usuario,
                )
            if respuesta:
                st.write("### Respuesta:")
                st.write(respuesta)
        else:
            st.warning("Por favor escribe una pregunta.")
else:
    st.info("Selecciona tu carrera para comenzar.")
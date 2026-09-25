import streamlit as st
import tiktoken
from groq import Groq
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

st.set_page_config(page_title="NLP & Groq Generative App", layout="wide")

st.title("Plataforma de Procesamiento de Lenguaje Natural y Generación")

# 1. Ingreso de API Key
st.sidebar.header("Configuración")
api_key = st.sidebar.text_input("Ingresa tu API Key de Groq:", type="password")

if not api_key:
    st.warning("Por favor, ingresa tu API Key de Groq en la barra lateral para comenzar.")
    st.stop()

# Inicializar cliente Groq
try:
    client = Groq(api_key=api_key)
except Exception as e:
    st.error(f"Error al inicializar cliente Groq: {e}")
    st.stop()

# Crear pestañas para las diferentes funcionalidades
tab1, tab2, tab3, tab4 = st.tabs([
    "Tokenización (GPT)", 
    "Bag of Words", 
    "Similitud Coseno", 
    "Esquema Generativo"
])

# 2. Mostrar tokens ID y tokens con colores (Usando tiktoken - GPT)
with tab1:
    st.header("Análisis de Tokens (Codificación GPT)")
    text_input = st.text_area("Ingresa el texto a tokenizar:", "El procesamiento de lenguaje natural es fascinante.")
    
    if text_input:
        # Usamos cl100k_base que es el tokenizer de GPT-3.5/GPT-4
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text_input)
        
        st.subheader("IDs de los Tokens:")
        st.write(tokens)
        
        st.subheader("Tokens con colores:")
        
        # Colores pastel para visualización
        colors = ["#ffadad", "#ffd6a5", "#fdffb6", "#caffbf", "#9bf6ff", "#a0c4ff", "#bdb2ff", "#ffc6ff"]
        
        html_content = "<div style='line-height: 2; font-size: 18px;'>"
        for i, token_id in enumerate(tokens):
            # Decodificar el token individual (manejando bytes)
            token_str = enc.decode_single_token_bytes(token_id).decode('utf-8', errors='replace')
            color = colors[i % len(colors)]
            html_content += f"<span style='background-color: {color}; padding: 2px 6px; border-radius: 4px; margin: 2px; color: black;'>{token_str}</span>"
        html_content += "</div>"
        
        st.markdown(html_content, unsafe_allow_html=True)

# 3. Mostrar Bag of Words
with tab2:
    st.header("Bag of Words (Bolsa de Palabras)")
    corpus_input = st.text_area("Ingresa varias oraciones (una por línea):", 
                                "El gato come pescado\nEl perro juega con la pelota\nEl gato y el perro son amigos")
    
    if corpus_input:
        corpus = [line.strip() for line in corpus_input.split('\n') if line.strip()]
        if corpus:
            vectorizer = CountVectorizer()
            X = vectorizer.fit_transform(corpus)
            
            df_bow = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
            st.write("Matriz de Frecuencias:")
            st.dataframe(df_bow)
        else:
            st.info("Ingresa al menos una oración.")

# 4. Mostrar Similitud con Distancia de Coseno
with tab3:
    st.header("Similitud de Coseno entre frases")
    phrase1 = st.text_input("Frase 1:", "Me encanta programar en Python")
    phrase2 = st.text_input("Frase 2:", "Disfruto mucho escribir código en Python")
    
    if phrase1 and phrase2:
        vectorizer_cos = CountVectorizer()
        vectors = vectorizer_cos.fit_transform([phrase1, phrase2])
        similarity = cosine_similarity(vectors)
        
        st.write(f"**Similitud de Coseno:** `{similarity[0][1]:.4f}`")
        
        # Interpretación visual
        st.progress(float(similarity[0][1]))
        if similarity[0][1] > 0.7:
            st.success("¡Las frases son muy similares!")
        elif similarity[0][1] > 0.3:
            st.warning("Las frases tienen similitud moderada.")
        else:
            st.info("Las frases son muy diferentes.")

# 5. Esquema Generativo (Modelos Groq - NO Llama)
with tab4:
    st.header("Generación de Texto con Groq")
    
    # Modelos disponibles en Groq que NO son Llama (ej. Mixtral, Gemma)
    # Nota: Groq no aloja GPT. Usamos alternativas open-source alojadas allí.
    available_models = [
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
        "gemma-7b-it"
    ]
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Parámetros")
        selected_model = st.selectbox("Selecciona el modelo (No-Llama):", available_models)
        temperature = st.slider("Temperatura", min_value=0.0, max_value=2.0, value=0.7, step=0.1, help="Mayor valor = más creatividad/aleatoriedad")
        max_tokens = st.slider("Max Tokens", min_value=50, max_value=4096, value=512, step=50)
        top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=1.0, step=0.05, help="Controla la diversidad del núcleo")
        
        # Nota: "Learning Rate" es un parámetro de entrenamiento, no de inferencia/generación. 
        # Se utilizan Temperature y Top P en su lugar para influir en la salida.
        
    with col2:
        st.subheader("Prompt")
        user_prompt = st.text_area("Escribe tu instrucción:", "Explica qué es la computación cuántica en un párrafo corto.", height=150)
        
        if st.button("Generar Respuesta", type="primary"):
            if user_prompt:
                with st.spinner(f"Generando respuesta usando {selected_model}..."):
                    try:
                        chat_completion = client.chat.completions.create(
                            messages=[
                                {
                                    "role": "user",
                                    "content": user_prompt,
                                }
                            ],
                            model=selected_model,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            top_p=top_p
                        )
                        st.write("### Respuesta:")
                        st.write(chat_completion.choices[0].message.content)
                    except Exception as e:
                        st.error(f"Ocurrió un error con la API: {e}")
            else:
                st.warning("Por favor, escribe un prompt primero.")

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import platform

# Mostrar versión de Python
st.write("Versión de Python:", platform.python_version())

# Cargar el modelo de Keras
try:
    model = load_model('keras_model.h5', compile=False)
    st.write("Modelo cargado exitosamente.")
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

# Configurar la interfaz de usuario
st.title("Reconocimiento de Imágenes")
st.sidebar.subheader("Usando un modelo entrenado en Teachable Machine puedes usar esta app para identificar imágenes")

# Mostrar imagen de referencia
image = Image.open('OIG5.jpg')
st.image(image, width=350)

# Capturar imagen desde la cámara
img_file_buffer = st.camera_input("Toma una Foto")

if img_file_buffer is not None:
    st.write("Imagen capturada exitosamente")
    
    try:
        # Leer imagen y preprocesarla
        img = Image.open(img_file_buffer).resize((224, 224))
        img_array = np.array(img)
        
        # Normalizar la imagen (asegurar que coincide con el modelo)
        normalized_image_array = (img_array.astype(np.float32) / 255.0)  # Ajustar si el modelo espera otra escala
        
        # Crear array para el modelo
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array
        
        # Realizar predicción
        prediction = model.predict(data)
        st.write("Predicciones:", prediction)
        
        # Etiquetas de clases (ajustar según el modelo)
        clases = ['Izquierda', 'Arriba', 'Derecha']  # Modifica si hay más o menos clases
        
        for i, clase in enumerate(clases):
            if i < len(prediction[0]) and prediction[0][i] > 0.5:
                st.header(f'{clase}, con Probabilidad: {prediction[0][i]:.2f}')
    
    except Exception as e:
        st.error(f"Error al procesar la imagen: {e}")

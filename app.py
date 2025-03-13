import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model

st.title("Reconocimiento de Imágenes")

# Subir imagen en vez de cargar una fija
img_file_buffer = st.file_uploader("Sube una imagen", type=["jpg", "png"])
if img_file_buffer is not None:
    img = Image.open(img_file_buffer)
    st.image(img, width=350)

    # Preprocesar imagen
    img = img.resize((224, 224))
    img_array = np.array(img)
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Hacer predicción
    if model_file is not None:
        prediction = model.predict(data)
        st.write("Predicción:", prediction)

if img_file_buffer is not None:
    # To read image file buffer with OpenCV:
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
   #To read image file buffer as a PIL Image:
    img = Image.open(img_file_buffer)

    newsize = (224, 224)
    img = img.resize(newsize)
    # To convert PIL Image to numpy array:
    img_array = np.array(img)

    # Normalize the image
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    print(prediction)
    if prediction[0][0]>0.5:
      st.header('Izquierda, con Probabilidad: '+str( prediction[0][0]) )
    if prediction[0][1]>0.5:
      st.header('Arriba, con Probabilidad: '+str( prediction[0][1]))
    #if prediction[0][2]>0.5:
    # st.header('Derecha, con Probabilidad: '+str( prediction[0][2]))



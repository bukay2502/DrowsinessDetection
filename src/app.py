import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load model
model_1 = tf.keras.models.load_model('DrowsinessDetection/src/model/drowsinessDetection.h5')
model_2 = tf.keras.models.load_model('DrowsinessDetection/src/model/drowsinessDetectionMobileNet.h5')

# Model selection
selected_model = st.radio("Select Model", ("Drowsiness Detection CNN", "Drowsiness Detection MobileNet"))

if selected_model == "Drowsiness Detection CNN":
    model = model_1
else:
    model = model_2

# Judul aplikasi
st.title('Deteksi Kantuk')

# Upload gambar
uploaded_file = st.file_uploader("Upload Gambar Wajah", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Baca gambar menggunakan PIL
    img = Image.open(uploaded_file)

    # Preprocessing
    if model == model_1:
        img = img.resize((112, 112))  # Ubah ukuran sesuai input model
    else:
        img = img.resize((128, 128))
    img = np.array(img) / 255.0  # Normalisasi
    img = np.expand_dims(img, axis=0)  # Tambah dimensi batch

    # Prediksi
    prediction = model.predict(img)

    # Tampilkan hasil
    st.image(img[0], caption='Gambar Input', use_container_width=True)
    if prediction[0][0] > 0.5:
        st.write("Prediksi: **Mengantuk**")
    else:
        st.write("Prediksi: **Tidak Mengantuk**")

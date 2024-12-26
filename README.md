# Klasifikasi Drowsiness dengan CNN dan MobileNet

## Deskripsi

Proyek ini bertujuan untuk mengembangkan sistem yang mampu mengklasifikasikan gambar wajah pengemudi untuk mendeteksi kondisi _drowsiness_ (mengantuk) menggunakan model Convolutional Neural Network (CNN) dan MobileNetV2. Sistem ini dilatih menggunakan dataset citra wajah yang didapatkan dari [Link Berikut](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd) yang telah dibagi menjadi dua kelas: "Drowsy" dan "Non_Drowsy".

## Model CNN

Model CNN yang digunakan dalam proyek ini memiliki arsitektur sebagai berikut:

*   **Lapisan Input:** Menerima gambar dengan ukuran 112x112 piksel dan 3 channel warna (RGB).
*   **Lapisan Konvolusi 1:** 32 filter dengan ukuran kernel 3x3, fungsi aktivasi ReLU.
*   **Lapisan Max Pooling 1:** Ukuran pool 2x2.
*   **Lapisan Flatten:** Mengubah output dari lapisan sebelumnya menjadi vektor satu dimensi.
*   **Lapisan Dense 1:** 64 neuron, fungsi aktivasi ReLU.
*   **Lapisan Dropout:**  Dropout rate 0.5.
*   **Lapisan Dense 2:** 2 neuron (sesuai dengan jumlah kelas), fungsi aktivasi softmax.

Model CNN ini dilatih menggunakan optimizer Adam dengan fungsi loss categorical crossentropy dan metrik evaluasi akurasi.

## Model MobileNetV2

Selain model CNN custom, proyek ini juga mengimplementasikan model MobileNetV2. MobileNetV2 adalah model CNN yang dirancang khusus untuk perangkat mobile dan embedded. Model ini memiliki ukuran yang lebih kecil dan komputasi yang lebih efisien dibandingkan dengan model CNN tradisional, sehingga cocok untuk dijalankan pada perangkat dengan sumber daya terbatas.

Arsitektur Model MobileNetV2:

*   Menggunakan MobileNetV2 yang telah dilatih sebelumnya dengan dataset ImageNet sebagai _base model_.
*   Lapisan klasifikasi pada _base model_ dihilangkan dan digantikan dengan lapisan `GlobalAveragePooling2D`, `Dropout(0.7)`, dan `Dense(2, activation='softmax')` untuk menyesuaikan dengan tugas klasifikasi biner.

## Dataset

Dataset yang digunakan dalam proyek ini berjumlah 41.793 citra. Dataset ini terdiri dari gambar wajah yang telah diklasifikasikan ke dalam dua kelas, yaitu "Drowsy" dan "Non_Drowsy". Dataset ini dibagi menjadi data training, validation, dan testing.

## Preprocessing

Sebelum dilatih, gambar-gambar dalam dataset di-preprocess terlebih dahulu. Proses preprocessing meliputi:

*   **Resizing:** Mengubah ukuran gambar menjadi 112x112 piksel (untuk model CNN) dan 128x128 piksel (untuk model MobileNetV2).
*   **Normalization:** Menormalisasi nilai piksel gambar ke rentang 0-1.
*   **Augmentation:** Melakukan augmentasi data pada data training untuk meningkatkan variasi data dan mencegah overfitting. Teknik augmentasi yang digunakan antara lain rotasi, flipping horizontal.

## Hasil

*   **Model CNN:** 
    *   Akurasi pada data testing: 99.95% 
    *   Presisi, recall, dan F1-score untuk kedua kelas adalah 1.00.
*   **Model MobileNetV2:**
    *   Akurasi pada data testing: 94.91%
    *   Presisi, recall, dan F1-score untuk kedua kelas di atas 90%.

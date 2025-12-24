# 🧥 Klasifikasi Citra Jenis Pakaian  
### Menggunakan Pendekatan Machine Learning

---

## 📌 Deskripsi Proyek

Proyek ini mengembangkan sistem **klasifikasi citra jenis pakaian** berbasis *machine learning* untuk mengenali dan mengelompokkan kategori pakaian secara otomatis dari citra digital. Sistem dibangun menggunakan pendekatan **Convolutional Neural Network (CNN)** dengan membandingkan model yang dilatih dari awal (*non-pretrained*) dan model *pretrained* berbasis **transfer learning**.

Model utama yang digunakan meliputi **CNN non-pretrained** sebagai model dasar, serta dua arsitektur *pretrained* yaitu **MobileNetV2** dan **EfficientNetB0**. Seluruh model diterapkan pada citra produk fashion dari platform e-commerce yang memiliki karakteristik *real-world images*, sehingga sistem diharapkan mampu bekerja secara representatif pada kondisi penggunaan nyata.

---

## 🎯 Tujuan Proyek

Tujuan dari proyek ini adalah membangun sistem **klasifikasi citra jenis pakaian** yang mampu mengenali kategori pakaian secara otomatis dari gambar digital. Sistem ini ditujukan untuk mendukung kebutuhan teknologi modern seperti **e-commerce**, **pengelolaan katalog produk**, dan **sistem rekomendasi berbasis visual**, serta menjadi fondasi pengembangan aplikasi *computer vision* yang membutuhkan efisiensi dan akurasi dalam pemrosesan citra.

---

## 📂 Dataset yang Digunakan

Dataset yang digunakan dalam proyek ini adalah **Zalando Store Crawl Dataset**, yang diperoleh dari platform Kaggle.

- **Nama Dataset** : Zalando Store Crawl  
- **Sumber** : Kaggle  
- **Link Dataset** :  
  https://www.kaggle.com/datasets/dqmonn/zalando-store-crawl

### 📊 Karakteristik Dataset

- **Jenis data** : Citra (image) pakaian  
- **Format gambar** : RGB  
- **Domain** : Fashion / Clothing  
- **Sumber gambar** : Produk fashion dari situs e-commerce Zalando  
- **Kondisi gambar** : *Real-world product images*  
- **Tujuan penggunaan** : Klasifikasi jenis pakaian  

---

## 🗂️ Struktur dan Seleksi Dataset

Dataset asli memiliki beberapa kelas dan variasi jumlah gambar per kelas.  
Dalam proyek ini dilakukan **seleksi dan penyederhanaan dataset** dengan ketentuan sebagai berikut:

### 📌 Kelas yang Digunakan

- Hoodies  
- Longsleeve  
- Shirt  
- Sweatshirt  

Kelas yang bersifat spesifik gender (*female*) **tidak digunakan** untuk menjaga konsistensi label dan menghindari bias dalam proses klasifikasi.

---

## 📊 Sampling Dataset

Untuk menjaga **keseimbangan data antar kelas**, dilakukan proses **sampling** dengan ketentuan:

- Setiap kelas diambil sebanyak **1.500 gambar**
- Pemilihan gambar dilakukan secara acak
- Total dataset yang digunakan:  
  **4 kelas × 1.500 gambar = 6.000 gambar**

Pendekatan ini bertujuan untuk:
- Menghindari ketidakseimbangan kelas (*class imbalance*)
- Mengurangi bias model terhadap kelas tertentu
- Menjaga efisiensi proses pelatihan

---

## 🗂️ Struktur Folder Dataset

Dataset disusun dalam bentuk **folder per kelas**, di mana setiap folder merepresentasikan satu kategori pakaian.  
Nama folder digunakan secara langsung sebagai **label kelas** pada proses pelatihan model.

---

## ⚙️ Preprocessing dan Pembagian Dataset

### 🔀 Pembagian Dataset (Train, Validation, Test)

Dataset yang telah diseleksi dan diseimbangkan kemudian dibagi ke dalam tiga subset, yaitu **training**, **validation**, dan **testing**. Pembagian dilakukan secara acak dengan rasio sebagai berikut:

- **Training** : 70%  
- **Validation** : 15%  
- **Testing** : 15%  

Proses pembagian dataset dilakukan secara otomatis menggunakan skrip Python dengan memastikan bahwa setiap kelas memiliki proporsi data yang sama pada masing-masing subset.

#### 📊 Distribusi Data per Kelas

| Subset     | Jumlah Gambar per Kelas |
|------------|--------------------------|
| Train      | 1.050 gambar             |
| Validation | 225 gambar               |
| Test       | 225 gambar               |

Total data yang digunakan tetap seimbang untuk keempat kelas pakaian, sehingga dapat meminimalkan risiko *class imbalance* selama proses pelatihan dan evaluasi model.

---

### 🖼️ Preprocessing Citra

Proses preprocessing citra dilakukan dengan menyesuaikan jenis model yang digunakan, yaitu antara **model CNN non-pretrained** dan **model pretrained berbasis transfer learning**.

#### 🔹 Preprocessing untuk CNN Non-Pretrained

Pada model CNN yang dilatih dari awal, preprocessing citra dilakukan menggunakan `ImageDataGenerator` dari TensorFlow/Keras dengan tahapan berikut:

- Resize gambar ke ukuran **224 × 224 piksel**
- Normalisasi nilai piksel ke rentang **[0, 1]**
- Label dikodekan dalam format **categorical**

Selain itu, diterapkan **augmentasi data** pada data training untuk meningkatkan kemampuan generalisasi model, meliputi:
- Rotasi gambar  
- Perubahan posisi horizontal dan vertikal  
- Zoom in/out  
- Horizontal flip  

Augmentasi **hanya diterapkan pada data training**, sedangkan data validation dan test hanya melalui proses normalisasi tanpa augmentasi.

---

#### 🔹 Preprocessing untuk Model Transfer Learning

Pada model pretrained, preprocessing citra disesuaikan dengan fungsi preprocessing bawaan dari masing-masing arsitektur. Sebagai contoh:

- **MobileNetV2** memetakan nilai piksel ke rentang **[-1, 1]**
- **EfficientNetB0** menggunakan skema normalisasi dan scaling khusus sesuai standar ImageNet

Pendekatan ini diimplementasikan menggunakan fungsi `preprocess_input` dari TensorFlow/Keras untuk memastikan kesesuaian distribusi data dengan proses pretraining.

---

### 🏷️ Mapping Kelas

Label kelas dihasilkan secara otomatis berdasarkan struktur folder dataset.  
Mapping kelas disimpan dalam format **JSON** untuk menjaga konsistensi label antara proses pelatihan dan implementasi sistem.

---

## 🧠 Model yang Digunakan

Proyek ini menggunakan tiga model *deep learning* untuk klasifikasi citra jenis pakaian:

- **CNN Non-Pretrained**  
  Model CNN yang dilatih dari awal tanpa bobot pretrained dan digunakan sebagai **baseline**.

- **MobileNetV2**  
  Model pretrained yang ringan dan efisien dengan pendekatan **transfer learning**.

- **EfficientNetB0**  
  Model pretrained dengan arsitektur modern yang dioptimalkan untuk keseimbangan akurasi dan kompleksitas model.

---

## 📈 Hasil dan Analisis

### 🔍 Ringkasan Hasil Evaluasi

Evaluasi performa model dilakukan menggunakan data uji (**test set**) dengan metrik **precision**, **recall**, **f1-score**, dan **accuracy**.

- CNN non-pretrained mampu mempelajari pola dasar pada citra pakaian, namun menunjukkan keterbatasan dalam membedakan kelas dengan karakteristik visual yang mirip.
- MobileNetV2 memberikan peningkatan performa yang signifikan berkat pemanfaatan fitur visual hasil *pretraining* ImageNet.
- EfficientNetB0 menghasilkan performa terbaik secara keseluruhan dengan distribusi metrik yang lebih seimbang antar kelas.

---

### 📊 Tabel Perbandingan Performa Model

| Model                         | Accuracy | Precision (Avg) | Recall (Avg) | F1-Score (Avg) |
|------------------------------|----------|------------------|--------------|----------------|
| CNN Non-Pretrained           | 0.63     | 0.62             | 0.63         | 0.60           |
| MobileNetV2 (Pretrained)    | 0.75     | 0.75             | 0.75         | 0.75           |
| EfficientNetB0 (Pretrained) | 0.76     | 0.76             | 0.76         | 0.76           |

---

### 📝 Analisis Singkat

Hasil eksperimen menunjukkan bahwa model pretrained berbasis **transfer learning** secara konsisten mengungguli CNN yang dilatih dari awal. **EfficientNetB0** memberikan performa terbaik dengan arsitektur yang lebih efisien dan representasi fitur yang kuat, sehingga lebih efektif untuk tugas klasifikasi citra pakaian berbasis data dunia nyata.

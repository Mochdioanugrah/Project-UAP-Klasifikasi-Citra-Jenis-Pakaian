import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

# =========================================================
# PAGE CONFIG (KOMPATIBEL STREAMLIT LAMA)
# =========================================================
st.set_page_config(
    page_title="Smart Clothing Classification",
    layout="wide"
)

# =========================================================
# HEADER UTAMA
# =========================================================
st.title("👕 SMART CLOTHING CLASSIFICATION")
st.caption("Mengenali jenis pakaian dari gambar menggunakan Deep Learning")

st.markdown(
    """
    Aplikasi ini memungkinkan siapa saja untuk **mengenali jenis pakaian secara otomatis**
    hanya dengan mengunggah gambar.  
    Tidak perlu pengetahuan teknis — cukup **pilih model, upload gambar, dan lihat hasilnya**.
    """
)

st.markdown("---")

# =========================================================
# MODEL CONFIG
# =========================================================
MODEL_CONFIGS = {
    "CNN Non-Pretrained": {
        "path": "Model/cnn_non_pretrained",
        "desc": "Model dasar (baseline) tanpa pretraining."
    },
    "MobileNetV2 Pretrained": {
        "path": "Model/mobilenetv2_pretrained",
        "desc": "Model ringan dan cepat, cocok untuk aplikasi nyata."
    },
    "EfficientNetB0 Pretrained": {
        "path": "Model/model_efficientnetb0",
        "desc": "Model modern dengan akurasi dan efisiensi tinggi."
    }
}

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model_assets(model_dir):
    model = tf.keras.models.load_model(
        f"{model_dir}/model.h5",
        compile=False
    )

    with open(f"{model_dir}/class_mapping.json") as f:
        class_mapping = json.load(f)

    with open(f"{model_dir}/config.json") as f:
        config = json.load(f)

    index_to_class = {v: k for k, v in class_mapping.items()}
    return model, index_to_class, config

# =========================================================
# SIDEBAR (PADAT & EDUKATIF)
# =========================================================
st.sidebar.header("🧭 Cara Menggunakan")

st.sidebar.markdown(
    """
    **1️⃣ Pilih Model AI**  
    Tentukan model klasifikasi yang ingin digunakan.

    **2️⃣ Upload Gambar**  
    Masukkan foto pakaian (JPG / PNG).

    **3️⃣ Lihat Hasil**  
    Sistem akan menampilkan jenis pakaian & keyakinan model.
    """
)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Model AI")

selected_model_name = st.sidebar.selectbox(
    "Pilih Model",
    list(MODEL_CONFIGS.keys())
)

st.sidebar.info(MODEL_CONFIGS[selected_model_name]["desc"])

model_dir = MODEL_CONFIGS[selected_model_name]["path"]
model, index_to_class, config = load_model_assets(model_dir)

st.sidebar.success(f"✅ Model aktif:\n{selected_model_name}")

with st.sidebar.expander("ℹ️ Info Teknis (Opsional)"):
    for k, v in config.items():
        st.write(f"**{k}** : {v}")

# =========================================================
# PREPROCESS
# =========================================================
def preprocess_image(image, config):
    img = image.resize((config["img_width"], config["img_height"]))
    img_array = np.expand_dims(np.array(img), axis=0)

    preprocessing = config.get("preprocessing_function", "rescale")

    if preprocessing == "mobilenet_v2.preprocess_input":
        img_array = mobilenet_preprocess(img_array)
    elif preprocessing == "efficientnet.preprocess_input":
        img_array = efficientnet_preprocess(img_array)
    else:
        img_array = img_array / 255.0

    return img_array

# =========================================================
# MAIN CONTENT – UPLOAD & INFO
# =========================================================
left, right = st.columns([2, 1])

with left:
    st.subheader("📤 Upload Gambar Pakaian")
    st.write(
        "Unggah **satu atau beberapa gambar pakaian**. "
        "Pastikan gambar cukup jelas agar hasil prediksi optimal."
    )

    uploaded_files = st.file_uploader(
        "Format didukung: JPG, JPEG, PNG",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

with right:
    st.subheader("🧠 Cara Kerja Sistem")
    st.markdown(
        """
        - 📷 Membaca gambar pakaian  
        - 🧮 Diproses oleh CNN  
        - 🎯 Menentukan jenis pakaian  
        - 📊 Menghitung tingkat keyakinan  
        """
    )

# =========================================================
# HASIL PREDIKSI
# =========================================================
if uploaded_files:
    st.markdown("---")
    st.subheader("🔍 Hasil Analisis")

    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")

        input_tensor = preprocess_image(image, config)
        predictions = model.predict(input_tensor, verbose=0)[0]

        idx = np.argmax(predictions)
        label = index_to_class[idx]
        confidence = predictions[idx] * 100

        st.markdown("----")
        col_img, col_main, col_prob = st.columns([1, 1.1, 1])

        # Gambar
        with col_img:
            st.image(
                image,
                caption=uploaded_file.name,
                use_column_width=True
            )

        # Hasil utama
        with col_main:
            st.success(f"👕 {label}")
            st.write(f"🎯 Tingkat Keyakinan: **{confidence:.2f}%**")

            if confidence >= 80:
                st.info("✅ Hasil sangat meyakinkan")
            elif confidence >= 60:
                st.warning("⚠️ Cukup meyakinkan")
            else:
                st.error("❗ Keyakinan rendah")

        # Probabilitas
        with col_prob:
            st.subheader("📊 Probabilitas Kelas")

            for i, p in enumerate(predictions):
                st.write(
                    f"- **{index_to_class[i]}** : {p*100:.2f}%"
                )

else:
    st.markdown("---")
    st.info("⬆️ Upload gambar untuk mulai menggunakan aplikasi ini.")

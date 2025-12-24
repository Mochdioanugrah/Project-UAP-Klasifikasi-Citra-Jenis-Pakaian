import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Clothing Classification Dashboard",
    layout="centered"
)

st.title("👕 Klasifikasi Jenis Pakaian")
st.caption("CNN Non-Pretrained vs MobileNetV2 vs EfficientNetB0")

# =========================================================
# MODEL CONFIGURATION
# =========================================================
MODEL_CONFIGS = {
    "CNN Non-Pretrained": "Model/cnn_non_pretrained",
    "MobileNetV2 Pretrained": "Model/mobilenetv2_pretrained",
    "EfficientNetB0 Pretrained": "Model/model_efficientnetb0"
}

# =========================================================
# LOAD MODEL & ASSETS
# =========================================================
@st.cache_resource
def load_model_assets(model_dir):
    model = tf.keras.models.load_model(
        f"{model_dir}/model.h5",
        compile=False
    )

    with open(f"{model_dir}/class_mapping.json", "r") as f:
        class_mapping = json.load(f)

    with open(f"{model_dir}/config.json", "r") as f:
        config = json.load(f)

    index_to_class = {v: k for k, v in class_mapping.items()}
    return model, index_to_class, config

# =========================================================
# MODEL SELECTOR
# =========================================================
selected_model_name = st.selectbox(
    "🔽 Pilih Model",
    list(MODEL_CONFIGS.keys())
)

model_dir = MODEL_CONFIGS[selected_model_name]
model, index_to_class, config = load_model_assets(model_dir)

st.success(f"Model aktif: {selected_model_name}")

# =========================================================
# INFO MODEL (NILAI TAMBAH UAP)
# =========================================================
with st.expander("ℹ️ Informasi Model"):
    for key, value in config.items():
        st.write(f"**{key}** : {value}")

# =========================================================
# PREPROCESSING FUNCTION (CONFIG-DRIVEN)
# =========================================================
def preprocess_image(image, config):
    img = image.resize((config["img_width"], config["img_height"]))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    preprocessing = config.get("preprocessing_function", "rescale")

    if preprocessing == "mobilenet_v2.preprocess_input":
        img_array = mobilenet_preprocess(img_array)

    elif preprocessing == "efficientnet.preprocess_input":
        img_array = efficientnet_preprocess(img_array)

    else:
        img_array = img_array / 255.0

    return img_array

# =========================================================
# MULTI IMAGE UPLOAD
# =========================================================
uploaded_files = st.file_uploader(
    "📤 Upload satu atau beberapa gambar pakaian",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    st.markdown("## 🔍 Hasil Prediksi")

    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")

        # Preprocess
        input_tensor = preprocess_image(image, config)

        # Predict
        predictions = model.predict(input_tensor)[0]
        predicted_index = np.argmax(predictions)
        predicted_label = index_to_class[predicted_index]
        confidence = predictions[predicted_index] * 100

        # Layout per gambar
        st.markdown("---")
        st.image(image, caption=f"File: {uploaded_file.name}", width=300)

        st.success(f"Prediksi: **{predicted_label}**")
        st.info(f"Confidence: **{confidence:.2f}%**")

        # Detail probabilitas
        with st.expander("📊 Lihat detail probabilitas"):
            for idx, prob in enumerate(predictions):
                st.write(f"{index_to_class[idx]} : {prob*100:.2f}%")

else:
    st.warning("Silakan upload satu atau beberapa gambar untuk diprediksi.")

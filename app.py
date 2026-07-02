import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

st.set_page_config(
    page_title="Drone Detection",
    layout="wide"
)

st.title("드론 탐지 시스템")
st.write("YOLO11 + OpenVINO 모델을 이용한 드론 탐지 앱입니다.")

MODEL_PATH = "drone_yolo11n_best_openvino_model"

@st.cache_resource
def load_model():
    model = YOLO(MODEL_PATH)
    return model

model = load_model()

st.sidebar.header("설정")
conf = st.sidebar.slider("신뢰도 기준", 0.10, 0.90, 0.25, 0.05)
imgsz = st.sidebar.selectbox("이미지 크기", [640, 768, 960], index=0)

uploaded_files = st.file_uploader(
    "이미지를 업로드하세요",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    st.write(f"업로드된 이미지 수: {len(uploaded_files)}장")

    for uploaded_file in uploaded_files:
        st.divider()
        st.subheader(uploaded_file.name)

        image = Image.open(uploaded_file).convert("RGB")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            temp_path = tmp.name

        results = model.predict(
            source=temp_path,
            imgsz=imgsz,
            conf=conf,
            device="cpu",
            verbose=False
        )

        result = results[0]
        result_img = result.plot()

        drone_count = len(result.boxes)

        col1, col2 = st.columns(2)

        with col1:
            st.write("원본 이미지")
            st.image(image, use_container_width=True)

        with col2:
            st.write("탐지 결과")
            st.image(result_img, use_container_width=True)

        st.success(f"탐지된 드론 수: {drone_count}개")

        os.remove(temp_path)

else:
    st.info("왼쪽 또는 위의 업로드 창에 드론 이미지를 넣어주세요.")
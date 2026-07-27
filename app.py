import streamlit as st
import cv2
from deep_translator import GoogleTranslator
from predict import predict_image, gemini_report

# ---------------- SESSION STATE ---------------- #
if "bone_image_path" not in st.session_state:
    st.session_state.bone_image_path = None
if "bone_result" not in st.session_state:
    st.session_state.bone_result = None
if "translated_text" not in st.session_state:
    st.session_state.translated_text = ""

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="MedAI | Bone Health & Translation",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ---------------- #
st.markdown("""
<style>
body {
    background-color: #f5f7fa;
}
.main {
    padding: 2rem;
}
.med-header {
    background: linear-gradient(90deg, #0f4c75, #3282b8);
    padding: 25px;
    border-radius: 12px;
    color: white;
    margin-bottom: 25px;
}
.med-card {
    background: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}
.med-title {
    color: #0f4c75;
    font-weight: 700;
}
.med-button button {
    background-color: #0f4c75 !important;
    color: white !important;
    border-radius: 8px !important;
}
.med-footer {
    text-align: center;
    color: gray;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ---------------- #
st.markdown("""
<div class="med-header">
    <h1>🩺 MedAI Healthcare Platform</h1>
    <p>AI-Powered Language Translation & Osteoporosis Detection</p>
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ---------------- #
st.sidebar.title("MedAI Navigation")
page = st.sidebar.radio(
    "Select Service",
    [" Language Translator", " Osteoporosis Analysis"]
)

# ---------------- LANGUAGE DATA ---------------- #
# Get supported languages directly from deep-translator
language_names = GoogleTranslator().get_supported_languages()

# ================= PAGE 1 ================= #
if page == " Language Translator":
    st.markdown("<h2 class='med-title'> Medical Language Translator</h2>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='med-card'>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            src_lang = st.selectbox("Source Language", language_names, index=language_names.index("english"))
            input_text = st.text_area("Enter Medical Text", height=180)

        with col2:
            dest_lang = st.selectbox("Target Language", language_names, index=language_names.index("hindi"))

        st.markdown("<div class='med-button'>", unsafe_allow_html=True)
        translate_btn = st.button(" Translate Text")
        st.markdown("</div>", unsafe_allow_html=True)

        if translate_btn:
            st.session_state.translated_text = GoogleTranslator(
                source=src_lang,
                target=dest_lang
            ).translate(input_text)

        if st.session_state.translated_text:
            st.text_area("Translated Output", st.session_state.translated_text, height=180)

        st.markdown("</div>", unsafe_allow_html=True)

# ================= PAGE 2 ================= #
elif page == " Osteoporosis Analysis":
    st.markdown("<h2 class='med-title'> Osteoporosis X-Ray Analysis</h2>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='med-card'>", unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload Bone X-Ray Image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file:
            with open("temp.jpg", "wb") as f:
                f.write(uploaded_file.read())

            st.session_state.bone_image_path = "temp.jpg"
            st.session_state.bone_result = predict_image("temp.jpg")

        if st.session_state.bone_image_path:
            col1, col2 = st.columns([1, 2])

            with col1:
                st.image(
                    st.session_state.bone_image_path,
                    caption="Uploaded X-Ray",
                    use_container_width=True
                )

            with col2:
                label, confidence = st.session_state.bone_result
                st.success(f" Diagnosis: **{label}**")
                st.info(f" Confidence Level: **{confidence:.2f}%**")

                if st.button(" Generate AI Medical Explanation"):
                    report = gemini_report(label, confidence)
                    st.markdown("###  AI Clinical Report")
                    st.write(report)

        st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ---------------- #
st.markdown("""
<div class="med-footer">
    <hr>
    <p> MedAI Healthcare | AI-Assisted Diagnosis & Communication</p>
    <p><small>Disclaimer: This tool is for educational purposes only.</small></p>
</div>
""", unsafe_allow_html=True)

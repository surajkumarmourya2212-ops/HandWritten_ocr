import streamlit as st
import numpy as np
from PIL import Image
import ocr_utils

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Handwritten OCR",
    page_icon="✍️",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------
@st.cache_resource
def load_models():
    digit_model = ocr_utils.load_digit_model()
    crnn_model = ocr_utils.load_crnn_model()
    return digit_model, crnn_model


digit_model, crnn_model = load_models()

# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.title("Handwritten OCR")

st.write(
    "Digit Model: **MNIST TensorFlow/Keras** | "
    "Text Engine: **Trained CRNN Model**"
)

# --------------------------------------------------
# TABS
# --------------------------------------------------
tab1, tab2 = st.tabs([
    "🔢 Digit Recognition",
    "📝 Text Recognition"
])

# ==================================================
# DIGIT RECOGNITION
# ==================================================
with tab1:

    st.header("🔢 Digit Recognition")

    st.write(
        "Upload an image containing a handwritten digit "
        "from 0 to 9."
    )

    digit_file = st.file_uploader(
        "📷 Upload handwritten digit",
        type=["png", "jpg", "jpeg", "webp"],
        key="digit_upload"
    )

    if digit_file is not None:

        image = Image.open(digit_file).convert("L")

        st.image(
            image,
            caption="Uploaded Digit",
            width=250
        )

        if st.button(
            "🔍 Recognize Digit",
            key="digit_button"
        ):

            image_array = np.array(image)

            try:
                prediction = ocr_utils.predict_digit(
                    image_array,
                    digit_model
                )

                st.success(
                    f"### Predicted Digit: **{prediction}**"
                )

            except Exception as e:

                st.error(
                    f"Digit recognition error: {e}"
                )


# ==================================================
# HANDWRITTEN TEXT RECOGNITION
# ==================================================
with tab2:

    st.header("📝 Handwritten Text Recognition")

    st.write(
        "Upload an image containing a handwritten "
        "word or short sentence."
    )

    st.info(
        "Text engine currently available: "
        "**Trained CRNN Model (model.h5)**"
    )

    text_file = st.file_uploader(
        "📷 Upload handwritten text",
        type=["png", "jpg", "jpeg", "webp"],
        key="text_upload"
    )

    if text_file is not None:

        image = Image.open(text_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Handwritten Text",
            use_container_width=True
        )

        if st.button(
            "🔍 Recognize Handwritten Text",
            key="text_button"
        ):

            image_array = np.array(image)

            with st.spinner(
                "Reading handwritten text..."
            ):

                try:

                    result = ocr_utils.predict_text(
                        image_array
                    )

                    if result and result.strip():

                        st.success("### Recognized Text")

                        st.text_area(
                            "OCR Result",
                            result,
                            height=120
                        )

                    else:

                        st.warning(
                            "No text could be recognized."
                        )

                except Exception as e:

                    st.error(
                        f"Text recognition error: {e}"
                    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")

st.caption(
    "Handwritten OCR | MNIST Digit Recognition + "
    "Trained CRNN Handwritten Text Recognition"
)

import streamlit as st
import numpy as np
from PIL import Image
import ocr_utils


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Handwritten OCR",
    page_icon="✍️",
    layout="wide"
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    digit_model = ocr_utils.load_digit_model()
    text_model = ocr_utils.load_crnn_model()

    return digit_model, text_model


try:

    digit_model, text_model = load_models()

except Exception as e:

    st.error("Model loading error")
    st.code(str(e))
    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title("Handwritten OCR")

st.write(
    "Digit Model: **MNIST TensorFlow/Keras** | "
    "Text Engine: **Trained CRNN Model**"
)


# ============================================================
# TABS
# ============================================================

digit_tab, text_tab = st.tabs(
    [
        "🔢 Digit Recognition",
        "📝 Text Recognition"
    ]
)


# ============================================================
# DIGIT RECOGNITION
# ============================================================

with digit_tab:

    st.header("🔢 Digit Recognition")

    st.write(
        "Upload an image containing one handwritten digit "
        "from 0 to 9."
    )

    digit_file = st.file_uploader(
        "📷 Upload handwritten digit",
        type=["png", "jpg", "jpeg", "webp"],
        key="digit_upload"
    )

    if digit_file is not None:

        try:

            digit_image = Image.open(
                digit_file
            ).convert("L")

            # Simple Streamlit 1.39 compatible image display
            st.image(
                digit_image,
                caption="Uploaded Digit"
            )

            if st.button(
                "🔍 Recognize Digit",
                key="digit_button"
            ):

                image_array = np.array(
                    digit_image
                )

                try:

                    # IMPORTANT:
                    # Only ONE argument is passed.
                    prediction_result = (
                        ocr_utils.predict_digit(
                            image_array
                        )
                    )

                    if isinstance(
                        prediction_result,
                        dict
                    ):

                        prediction = prediction_result.get(
                            "prediction",
                            ""
                        )

                        confidence = prediction_result.get(
                            "confidence",
                            None
                        )

                    else:

                        prediction = prediction_result
                        confidence = None


                    st.success(
                        f"Predicted Digit: **{prediction}**"
                    )


                    if confidence is not None:

                        st.write(
                            f"Confidence: "
                            f"**{confidence * 100:.2f}%**"
                        )

                except Exception as e:

                    st.error(
                        "Digit recognition error"
                    )

                    st.code(
                        str(e)
                    )

        except Exception as e:

            st.error(
                "Could not open digit image"
            )

            st.code(
                str(e)
            )


# ============================================================
# HANDWRITTEN TEXT RECOGNITION
# ============================================================

with text_tab:

    st.header(
        "📝 Handwritten Text Recognition"
    )

    st.write(
        "Upload an image containing a handwritten "
        "word or short sentence."
    )

    st.info(
        "Text engine: **Trained CRNN Model (`model.h5`)**"
    )


    text_file = st.file_uploader(
        "📷 Upload handwritten text",
        type=["png", "jpg", "jpeg", "webp"],
        key="text_upload"
    )


    if text_file is not None:

        try:

            text_image = Image.open(
                text_file
            ).convert("RGB")

            # Simple Streamlit-compatible display
            st.image(
                text_image,
                caption="Uploaded Handwritten Text"
            )


            if st.button(
                "🔍 Recognize Handwritten Text",
                key="text_button"
            ):

                image_array = np.array(
                    text_image
                )


                with st.spinner(
                    "Reading handwritten text..."
                ):

                    try:

                        result = (
                            ocr_utils.predict_text(
                                image_array
                            )
                        )


                        if isinstance(
                            result,
                            dict
                        ):

                            recognized_text = (
                                result.get(
                                    "prediction",
                                    ""
                                )
                            )

                            engine = (
                                result.get(
                                    "engine",
                                    "Trained CRNN Model"
                                )
                            )

                        else:

                            recognized_text = str(
                                result
                            )

                            engine = (
                                "Trained CRNN Model"
                            )


                        recognized_text = (
                            recognized_text.strip()
                        )


                        if recognized_text:

                            st.success(
                                "Recognized Text"
                            )

                            st.text_area(
                                "OCR Result",
                                recognized_text,
                                height=150
                            )

                            st.caption(
                                f"Engine: {engine}"
                            )

                        else:

                            st.warning(
                                "No handwritten text "
                                "was recognized."
                            )


                    except Exception as e:

                        st.error(
                            "Text recognition error"
                        )

                        st.code(
                            str(e)
                        )


        except Exception as e:

            st.error(
                "Could not open text image"
            )

            st.code(
                str(e)
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Handwritten OCR | "
    "MNIST Digit Recognition + "
    "Trained CRNN Handwritten Text Recognition"
)
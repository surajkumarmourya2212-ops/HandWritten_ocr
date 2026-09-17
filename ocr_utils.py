```python
import os
import numpy as np
import cv2
import tensorflow as tf

# ============================================================
# OPTIONAL IMPORTS
# ============================================================

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False


try:
    from mltu.transformers import ImageResizer
    from mltu.utils.text_utils import ctc_decoder
    MLTU_AVAILABLE = True
except ImportError:
    MLTU_AVAILABLE = False


# ============================================================
# MODEL PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Default paths
# Can also be overridden using environment variables.

DIGIT_MODEL_PATH = os.getenv(
    "DIGIT_MODEL_PATH",
    os.path.join(MODELS_DIR, "digit_model.keras")
)

CRNN_MODEL_PATH = os.getenv(
    "CRNN_MODEL_PATH",
    os.path.join(MODELS_DIR, "crnn_model.h5")
)


# ============================================================
# MODEL CACHE
# ============================================================

_digit_model = None
_crnn_model = None


# ============================================================
# LOAD DIGIT MODEL
# ============================================================

def load_digit_model():

    global _digit_model

    if _digit_model is not None:
        return _digit_model

    if not os.path.exists(DIGIT_MODEL_PATH):
        raise FileNotFoundError(
            f"Digit model not found: {DIGIT_MODEL_PATH}"
        )

    _digit_model = tf.keras.models.load_model(
        DIGIT_MODEL_PATH
    )

    return _digit_model


# ============================================================
# LOAD CRNN MODEL
# ============================================================

def load_crnn_model():

    global _crnn_model

    if _crnn_model is not None:
        return _crnn_model

    if not os.path.exists(CRNN_MODEL_PATH):
        return None

    try:

        _crnn_model = tf.keras.models.load_model(
            CRNN_MODEL_PATH,
            compile=False,
            safe_mode=False
        )

        return _crnn_model

    except Exception as e:

        print(
            "Could not load CRNN model:",
            e
        )

        return None


# ============================================================
# TESSERACT CHECK
# ============================================================

def tesseract_available():

    if not PYTESSERACT_AVAILABLE:
        return False

    try:

        pytesseract.get_tesseract_version()

        return True

    except Exception:

        return False


# ============================================================
# IMAGE HELPER
# ============================================================

def _to_gray(image):

    if len(image.shape) == 3:

        return cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )

    return image.copy()


# ============================================================
# DIGIT VALIDATION
# ============================================================

def is_valid_digit_image(image):

    gray = _to_gray(image)

    gray = cv2.resize(
        gray,
        (200, 200)
    )

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = [
        c for c in contours
        if cv2.contourArea(c) > 30
    ]

    if len(contours) == 0 or len(contours) > 5:
        return False

    largest = max(
        contours,
        key=cv2.contourArea
    )

    largest_area = cv2.contourArea(
        largest
    )

    total_area = sum(
        cv2.contourArea(c)
        for c in contours
    )

    if (
        total_area == 0
        or largest_area / total_area < 0.60
    ):
        return False

    x, y, w, h = cv2.boundingRect(
        largest
    )

    image_area = (
        binary.shape[0]
        * binary.shape[1]
    )

    size_ratio = (
        w * h
    ) / image_area

    if (
        size_ratio < 0.005
        or size_ratio > 0.75
    ):
        return False

    aspect_ratio = w / float(h)

    return (
        0.15
        <= aspect_ratio
        <= 5.0
    )


# ============================================================
# DIGIT PREPROCESSING
# ============================================================

def preprocess_digit_image(image):

    gray = _to_gray(image)

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    useful = [
        c for c in contours
        if cv2.contourArea(c) > 10
    ]

    if useful:

        largest = max(
            useful,
            key=cv2.contourArea
        )

        x, y, w, h = cv2.boundingRect(
            largest
        )

        padding = int(
            max(w, h) * 0.25
        )

        x1 = max(
            0,
            x - padding
        )

        y1 = max(
            0,
            y - padding
        )

        x2 = min(
            binary.shape[1],
            x + w + padding
        )

        y2 = min(
            binary.shape[0],
            y + h + padding
        )

        binary = binary[
            y1:y2,
            x1:x2
        ]

    h, w = binary.shape

    size = max(
        h,
        w
    )

    square = np.zeros(
        (size, size),
        dtype=np.uint8
    )

    y_offset = (
        size - h
    ) // 2

    x_offset = (
        size - w
    ) // 2

    square[
        y_offset:y_offset + h,
        x_offset:x_offset + w
    ] = binary

    resized = cv2.resize(
        square,
        (28, 28),
        interpolation=cv2.INTER_AREA
    )

    resized = (
        resized.astype("float32")
        / 255.0
    )

    model = load_digit_model()

    input_shape = model.input_shape

    if len(input_shape) == 4:

        return resized.reshape(
            1,
            28,
            28,
            1
        )

    elif len(input_shape) == 3:

        return resized.reshape(
            1,
            28,
            28
        )

    elif len(input_shape) == 2:

        return resized.reshape(
            1,
            784
        )

    else:

        raise ValueError(
            f"Unsupported model input shape: "
            f"{input_shape}"
        )


# ============================================================
# DIGIT PREDICTION
# ============================================================

def predict_digit(image):

    if not is_valid_digit_image(image):

        raise ValueError(
            "Invalid image. Please upload "
            "one clear handwritten digit (0-9)."
        )

    model = load_digit_model()

    processed = preprocess_digit_image(
        image
    )

    probabilities = model.predict(
        processed,
        verbose=0
    )[0]

    prediction = int(
        np.argmax(probabilities)
    )

    confidence = float(
        np.max(probabilities)
    )

    return {
        "prediction": prediction,
        "confidence": confidence,
        "engine": "MNIST TensorFlow/Keras Model"
    }


# ============================================================
# TEXT RECOGNITION
# TRAINED MLTU CRNN + CTC
# ============================================================

def predict_text_crnn(image):

    model = load_crnn_model()

    if model is None:
        return None

    if not MLTU_AVAILABLE:

        print(
            "MLTU is not installed."
        )

        return None

    try:

        # ----------------------------------------------------
        # Get dimensions directly from trained model
        # ----------------------------------------------------

        height = model.input_shape[1]
        width = model.input_shape[2]

        # ----------------------------------------------------
        # Same preprocessing used during training
        # ----------------------------------------------------

        image = ImageResizer.resize_maintaining_aspect_ratio(
            image,
            width,
            height
        )

        # ----------------------------------------------------
        # Add batch dimension
        # ----------------------------------------------------

        image = np.expand_dims(
            image,
            axis=0
        ).astype(np.float32)

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        predictions = model.predict(
            image,
            verbose=0
        )

        # ----------------------------------------------------
        # Exact vocabulary used during training
        # ----------------------------------------------------

        vocab = (
            'sXE-A5!krH+SMh93F?Pg,KYm)R/QzN:ne;'
            'VpjxuGO*TyLJl"vit0WB#w4Zq6aob1D.7d8UCc(&f2\'I'
        )

        # ----------------------------------------------------
        # MLTU CTC decoder
        # ----------------------------------------------------

        result = ctc_decoder(
            predictions,
            vocab
        )[0]

        return result.strip()

    except Exception as e:

        print(
            "CRNN prediction error:",
            e
        )

        return None


# ============================================================
# TEXT PREDICTION
# ============================================================

def predict_text(image):

    # --------------------------------------------------------
    # Try trained CRNN first
    # --------------------------------------------------------

    if load_crnn_model() is not None:

        prediction = predict_text_crnn(
            image
        )

        if prediction is not None:

            return {
                "prediction": prediction,
                "engine": "CRNN + CTC"
            }

    # --------------------------------------------------------
    # Tesseract fallback
    # --------------------------------------------------------

    gray = _to_gray(image)

    if tesseract_available():

        prediction = pytesseract.image_to_string(
            gray,
            config="--psm 6"
        )

        return {
            "prediction": prediction.strip(),
            "engine": "Tesseract OCR"
        }

    # --------------------------------------------------------
    # No engine available
    # --------------------------------------------------------

    raise RuntimeError(
        "No text recognition engine available. "
        "Add models/crnn_model.h5 or install "
        "Tesseract OCR."
    )
```

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import tensorflow as tf
import logging
import os

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model2.keras")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ MODEL FILE NOT FOUND at {MODEL_PATH}")

# ---------------- LOAD MODEL ----------------
MODEL = tf.keras.models.load_model(MODEL_PATH)

CLASSNAMES = ["Early Blight", "Healthy", "Late Blight"]
IMAGE_SIZE = 256

logger.info("✅ MODEL LOADED SUCCESSFULLY")
logger.info(f"Model input shape: {MODEL.input_shape}")
logger.info(f"Model output shape: {MODEL.output_shape}")
logger.info(f"Total parameters: {MODEL.count_params()}")

# ---------------- APP ----------------
app = FastAPI(title="Potato Disease Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- ROUTES ----------------
@app.get("/hello")
async def hello():
    return {"message": "Hello, World!"}


def read_file_as_image(data: bytes) -> np.ndarray:
    """
    IMPORTANT:
    Model already has Rescaling(1./255) inside,
    so DO NOT divide by 255 here.
    """
    image = tf.io.decode_image(data, channels=3, expand_animations=False)
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = tf.cast(image, tf.float32)
    return image.numpy()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = read_file_as_image(await file.read())
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    img_batch = np.expand_dims(image, axis=0)

    # Force inference mode
    predictions = MODEL(img_batch, training=False).numpy()

    logger.info(f"RAW MODEL OUTPUT: {predictions}")

    for i, class_name in enumerate(CLASSNAMES):
        logger.info(f"{class_name}: {predictions[0][i]:.4f}")

    predicted_index = int(np.argmax(predictions[0]))
    predicted_class = CLASSNAMES[predicted_index]
    confidence = float(predictions[0][predicted_index])

    logger.info(f"PREDICTED CLASS: {predicted_class}")
    logger.info(f"CONFIDENCE: {confidence:.4f}")

    return {
        "class": predicted_class,
        "confidence": confidence
    }


# ---------------- LOCAL RUN ----------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)

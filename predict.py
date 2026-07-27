import tensorflow as tf
import numpy as np
import cv2
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

groq_model = ChatGroq(model='llama-3.1-8b-instant')


# Load CNN model
model = tf.keras.models.load_model("model/bone_cnn.h5")

def predict_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Image path incorrect or image not found")

    img = cv2.resize(img, (224,224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]

    label = "Osteoporosis (Low Bone Density)" if pred > 0.5 else "Normal"
    confidence = pred if pred > 0.5 else 1 - pred

    return label, round(confidence * 100, 2)

def gemini_report(label, confidence):
    prompt = f"""
    You are a medical AI assistant.
    Explain this Osteoporosis analysis result in simple terms.
    Do NOT give diagnosis.

    Result:
    Condition: {label}
    Confidence: {confidence}%
    """

    response = groq_model.invoke(prompt)
    return response.text


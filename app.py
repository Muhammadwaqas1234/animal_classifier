# app.py: FastAPI application for Keras Image Classification (with CORS)

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware # <-- NEW IMPORT
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, Rescaling
from tensorflow.keras.applications import MobileNetV2
from PIL import Image
import numpy as np
import io
import pickle
import os

# --- Configuration ---
MODEL_WEIGHTS_PATH = "image_classifier_weights.weights.h5"
CLASS_NAMES_PATH = "class_names.pkl"
IMAGE_SIZE = (224, 224) 
# --- CORS Configuration (Allow requests from all origins for development) ---
origins = ["*"] 

# --- FastAPI Initialization ---
app = FastAPI(
    title="Keras Image Classifier API",
    description="Loads model architecture and weights separately and includes CORS.",
    version="1.0.3"
)

# --- Add CORS Middleware (NEW) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # Allows all origins during development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store the loaded model and class names
model = None
class_names = None

@app.on_event("startup")
async def load_resources():
    # ... (Model loading logic remains the same, as provided by you)
    global model, class_names
    
    # --- 1. Load Class Names ---
    if not os.path.exists(CLASS_NAMES_PATH):
        print(f"❌ ERROR: Class names file not found: {CLASS_NAMES_PATH}")
        return 
        
    try:
        with open(CLASS_NAMES_PATH, 'rb') as f:
            class_names = pickle.load(f)
        NUM_CLASSES = len(class_names)
        print(f"✅ Class names loaded: {class_names}")
    except Exception as e:
        print(f"❌ Error loading class names: {e}")
        raise RuntimeError(f"Failed to load class names: {e}")

    # --- 2. Rebuild the Model Architecture ---
    try:
        base_model = MobileNetV2(
            input_shape=IMAGE_SIZE + (3,),
            include_top=False,
            weights=None
        )
        base_model.trainable = True
        
        model = Sequential([
            Rescaling(1.0 / 255), 
            base_model, 
            GlobalAveragePooling2D(),
            Dropout(0.3),
            Dense(NUM_CLASSES, activation="softmax")
        ])
        
        model.build((None, *IMAGE_SIZE, 3)) 
        print("✅ Model architecture rebuilt successfully.")

        # --- 3. Load the Saved Weights ---
        if not os.path.exists(MODEL_WEIGHTS_PATH):
            print(f"❌ ERROR: Model weights file not found: {MODEL_WEIGHTS_PATH}")
            return
            
        model.load_weights(MODEL_WEIGHTS_PATH)
        print(f"✅ Model weights loaded from: {MODEL_WEIGHTS_PATH}")

    except Exception as e:
        print(f"❌ Error during model rebuild or weights loading: {e}")
        raise RuntimeError(f"Failed to load critical resources: {e}")


@app.get("/")
def home():
    """Root endpoint for basic health check."""
    if model is None:
        return {"status": "error", "message": "Model not loaded. Check server logs."}
    return {"status": "ok", "message": "Image Classifier API is running!"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to receive an image file and return the classification prediction.
    """
    if model is None or class_names is None:
        raise HTTPException(status_code=500, detail="Model or class names not loaded.")

    try:
        # Read and preprocess the image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize(IMAGE_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) 

        # Make Prediction
        predictions = model.predict(img_array, verbose=0)
        
        # Process results
        predicted_class_index = np.argmax(predictions[0]) 
        predicted_class_name = class_names[predicted_class_index]
        confidence = float(predictions[0][predicted_class_index])

        return {
            "filename": file.filename,
            "prediction": predicted_class_name,
            "confidence": f"{confidence:.4f}",
            "all_scores": dict(zip(class_names, [f"{p:.4f}" for p in predictions[0]]))
        }

    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image or making prediction: {e}")
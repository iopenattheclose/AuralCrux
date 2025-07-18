# backend/main.py

import base64
import io
import numpy as np
import requests # This might still be used by your AudioClassifier
import torch    # This might still be used by your AudioClassifier
import torch.nn as nn # For dummy AudioCNN
import torchaudio.transforms as T # For dummy AudioProcessor
import soundfile as sf
import librosa

# --- Import your core logic from inference_logic.py ---
from inference import AudioClassifier, predict 

# --- FastAPI imports ---
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# --- FastAPI App Setup ---
app = FastAPI(
    title="CNN Audio Visualizer Backend",
    description="API for uploading WAV files and getting audio analysis predictions.",
    version="0.1.0",
)

# --- CORS Configuration (Crucial for local development) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins during development (e.g., http://localhost:5173)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# --- Instantiate and Load Model Once at App Startup ---
# This ensures the model is loaded only when the FastAPI server starts,
# not on every incoming request.
audio_classifier_instance = AudioClassifier()
audio_classifier_instance.load_model() # Call load_model here to load your actual model assets

# --- FastAPI Endpoint ---
@app.post("/api/predict", summary="Uploads a WAV file and processes it")
async def process_uploaded_audio(audioFile: UploadFile = File(...)):
    """
    Receives an uploaded WAV file from the frontend, reads its bytes,
    and passes them to the `predict()` function for analysis.
    """
    print(f"Received file: {audioFile.filename}, Content-Type: {audioFile.content_type}")

    # 1. Validate file type
    if not audioFile.content_type == "audio/wav":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only WAV audio files are accepted."
        )

    try:
        # 2. Read the uploaded file's content bytes asynchronously
        uploaded_audio_bytes = await audioFile.read()
        print(f"File size received from frontend: {len(uploaded_audio_bytes)} bytes")

        # 3. Call your `predict()` function, passing the uploaded file's bytes
        #    This is where the actual prediction logic is triggered.
        prediction_result = predict(uploaded_audio_bytes,audio_classifier_instance)

        # 4. Return the result from predict() to the frontend
        return JSONResponse(content=prediction_result, status_code=status.HTTP_200_OK)

    except HTTPException as e:
        # Re-raise HTTPExceptions (e.g., from validation or if predict raises one)
        raise e
    except Exception as e:
        # Catch any other unexpected errors during file reading or prediction
        print(f"An unexpected error occurred during processing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal server error occurred: {str(e)}"
        )
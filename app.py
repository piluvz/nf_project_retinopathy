import os
import uvicorn
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from transformers import pipeline, ViTImageProcessor
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the pipeline
try:
    image_processor = ViTImageProcessor.from_pretrained("Kontawat/vit-diabetic-retinopathy-classification")
    pipe = pipeline("image-classification", model="Kontawat/vit-diabetic-retinopathy-classification", feature_extractor=image_processor)
except Exception as e:
    print(f"Error loading model: {e}")
    pipe = None

LABELS_MAP = {
    "0": "No Diabetic Retinopathy",
    "1": "Mild Diabetic Retinopathy",
    "2": "Moderate Diabetic Retinopathy",
    "3": "Severe Diabetic Retinopathy",
    "4": "Proliferative Diabetic Retinopathy"
}

EXPLANATIONS = {
    "0": "No signs of diabetic retinopathy detected. Your eyes appear healthy.",
    "1": "Mild diabetic retinopathy detected. Early signs of damage to the retina. Regular monitoring is recommended.",
    "2": "Moderate diabetic retinopathy detected. Some damage to the retina is present. Treatment may be necessary.",
    "3": "Severe diabetic retinopathy detected. Significant damage to the retina. Immediate treatment is required to prevent vision loss.",
    "4": "Proliferative diabetic retinopathy detected. Extensive damage to the retina with abnormal blood vessel growth. Urgent treatment is necessary."
}

@app.post("/classify-image")
async def classify_image(file: UploadFile = File(...)):
    if pipe is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Perform image classification
        results = pipe(image)
        
        # Get the highest confidence prediction
        highest_confidence_prediction = max(results, key=lambda x: x['score'])
        label = highest_confidence_prediction['label']
        description = {
            "label": LABELS_MAP[label],
            "score": highest_confidence_prediction['score'],
            "explanation": EXPLANATIONS[label]
        }

        return JSONResponse(content=description)

    except Exception as e:
        print(f"Error during classification: {e}")
        raise HTTPException(status_code=500, detail="Error processing image")

@app.get("/")
async def root():
    return {"message": "Welcome to the Image Classification API"}
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from services.translator import translate_to_aksara_jawa
from PIL import Image
from io import BytesIO
from utils.segmentation import multi_line_character_segmentation
from utils.restructure_text import reconstruct_javanese_text
from utils.hardcoded_fixes import hardcoded_fixes

# Initialize FastAPI
app = FastAPI(title="Javanese to Aksara Jawa Translator", version="1.0")


# Request Model
class TranslationRequest(BaseModel):
    text: str


# API Endpoint
@app.post("/translate")
async def translate(request: TranslationRequest):
    return translate_to_aksara_jawa(request.text)


@app.post("/ocr-hanacaraka")
async def predict(file: UploadFile = File(...)):
    """Receives an image, processes it, and returns OCR results."""
    try:
        # Read and convert uploaded file
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")

        # Perform segmentation & recognition
        ocr_results = multi_line_character_segmentation(image)

        # Apply segmentation and fuzzy reconstruction
        reconstructed_text = reconstruct_javanese_text(ocr_results)

        # ✅ Cek dan koreksi menggunakan hardcoded_fixes
        sentence = reconstructed_text["final_sentence"]
        if sentence in hardcoded_fixes:
            correction = hardcoded_fixes[sentence]
            reconstructed_text["reconstruct"] = {
                k: correction for k in reconstructed_text["reconstruct"].keys()
            }
            reconstructed_text["final_sentence"] = correction

        print(reconstructed_text["final_sentence"])

        return {
            "ocr_results": ocr_results,
            "reconstructed_text": reconstructed_text["reconstruct"],
            "final_sentence": reconstructed_text["final_sentence"],
        }

    except Exception as e:
        return {"error": str(e)}

import re
import google.generativeai as genai
from config import GEMINI_API_KEY

# Configure Gemini AI
genai.configure(api_key=GEMINI_API_KEY)


# Translation Function
def translate_to_aksara_jawa(javanese_text: str):
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")

        prompt = (
            "Translate the following Javanese phrase into pure Aksara Jawa script. "
            "Ensure that ALL words are translated completely, with proper spacing. "
            "Do NOT omit any words. Do NOT add any explanations. Only return the Aksara Jawa script:\n\n"
            f"Javanese: {javanese_text}\nAksara Jawa:"
        )

        response = model.generate_content(prompt)
        raw_response = response.text.strip()

        # Post-processing: Keep only Aksara Jawa characters
        aksara_jawa_only = re.sub(r"[^\uA980-\uA9DF\s]", "", raw_response)
        aksara_jawa_only = re.sub(r"\s+", " ", aksara_jawa_only).strip()

        return {
            "input": javanese_text,
            "raw_response": raw_response,
            "cleaned_translation": aksara_jawa_only,
            "debug": {
                "used_prompt": prompt,
                "model_used": "gemini-2.0-flash",
            },
        }

    except Exception as e:
        return {"error": str(e)}

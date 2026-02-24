import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pptx import Presentation
import google.generativeai as genai
import json

app = FastAPI()

# Configure Gemini
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-2.5-flash')

def extract_text_from_pptx(file_path):
    prs = Presentation(file_path)
    full_text = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                full_text.append(shape.text)
    return "\n".join(full_text)

@app.post("/generate-quiz")
async def generate_quiz(file: UploadFile = File(...)):
    if not file.filename.endswith('.pptx'):
        raise HTTPException(status_code=400, detail="Only .pptx files are supported.")

    # Save temp file to parse
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())

    try:
        # 1. Parse PPTX
        context_text = extract_text_from_pptx(temp_path)
        
        # 2. Construct Prompt
        prompt = f"""
        Extract the most important information from the following text and create a quiz.
        Requirements:
        - Exactly 20 Multiple Choice Questions.
        - 4 options per question.
        - Clearly indicate the correct answer.
        - Return the result strictly as a JSON list of objects.
        
        Text: {context_text[:10000]} 
        """

        # 3. Call Gemini
        response = model.generate_content(prompt)
        
        # Clean up the AI response (removing markdown code blocks if present)
        raw_text = response.text.replace('```json', '').replace('```', '').strip()
        quiz_data = json.loads(raw_text)

        return {"quiz": quiz_data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/")
def health_check():
    return {"status": "Service is running"}

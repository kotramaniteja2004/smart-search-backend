import json
from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from PIL import Image
import io
import os
import requests
import base64
import concurrent.futures

app = FastAPI(title="Cloud AI Search")

# Configuration
GOOGLE_API_KEY = "AIzaSyCVBozxMSHn7oNlYGc5nmqjVJ45rY8G3Uc"
BASE_IMAGE_FOLDER = "uploaded_images"
METADATA_FILE = "metadata.json"

class SearchRequest(BaseModel):
    query: str

# Helper to save/load metadata
def load_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r") as f:
            return json.load(f)
    return {}

def save_metadata(data):
    with open(METADATA_FILE, "w") as f:
        json.dump(data, f)

@app.post("/sync-photo")
async def sync_photo(user_id: str, file: UploadFile = File(...), photo_id: str = Form(...)):
    user_folder = os.path.join(BASE_IMAGE_FOLDER, user_id)
    os.makedirs(user_folder, exist_ok=True)
    
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image.thumbnail((800, 800))
    image_path = os.path.join(user_folder, f"{photo_id}.jpg")
    image.save(image_path, optimize=True, quality=85)
    
    # 🚀 CLOUD BRAIN: Ask Gemini to "memorize" what is in this photo
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    prompt = "Describe this image in 50 words for a search index. Include colors, objects, and any text/prices seen."
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GOOGLE_API_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": base64_image}}]}]}
    
    try:
        response = requests.post(url, json=payload).json()
        description = response['candidates'][0]['content']['parts'][0]['text']
        
        metadata = load_metadata()
        if user_id not in metadata: metadata[user_id] = {}
        metadata[user_id][photo_id] = description
        save_metadata(metadata)
        
        return {"status": "success", "description": description}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/synced-photos")
async def get_synced_photos(user_id: str):
    metadata = load_metadata()
    user_data = metadata.get(user_id, {})
    return {"synced_ids": list(user_data.keys())}

def verify_match(photo_id, description, query, user_id):
    # This function uses Gemini to check if the specific photo matches the user's search
    file_path = os.path.join(BASE_IMAGE_FOLDER, user_id, f"{photo_id}.jpg")
    with open(file_path, "rb") as img:
        base64_image = base64.b64encode(img.read()).decode("utf-8")

    prompt = f'User is searching for: "{query}". Does this photo description match? "{description}". Reply ONLY in JSON: {{"match": bool, "is_receipt": bool, "vendor": str, "total_price": str, "description": str}}'
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GOOGLE_API_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": base64_image}}]}]}
    
    try:
        resp = requests.post(url, json=payload).json()
        text = resp['candidates'][0]['content']['parts'][0]['text']
        data = json.loads(text.replace('```json', '').replace('
', '').strip())

        if not data.get('match', False): return None
        
        return {
            "image_found": f"{user_id}/{photo_id}", 
            "image_base64": base64_image,
            "is_receipt": data.get('is_receipt', False),
            "vendor": data.get('vendor', 'Unknown'),
            "total_price": data.get('total_price', '0'),
            "non_receipt_description": data.get('description', description)
        }
    except: return None

@app.post("/search-and-extract")
async def search_and_extract(user_id: str, request: SearchRequest):
    metadata = load_metadata()
    user_photos = metadata.get(user_id, {})
    
    if not user_photos:
        return {"error": "No photos synced yet!"}

    # Narrow down potential matches by simple keyword check first to save API costs
    query_words = request.query.lower().split()
    potential_ids = []
    for pid, desc in user_photos.items():
        if any(word in desc.lower() for word in query_words):
            potential_ids.append((pid, desc))
    
    # If no keywords match, try the top 5 most recent anyway
    if not potential_ids:
        potential_ids = list(user_photos.items())[-5:]

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(verify_match, pid, desc, request.query, user_id) for pid, desc in potential_ids[:10]]
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            if res: results.append(res)

    if not results:
        return {"error": f"No matches found for '{request.query}'"}
        
    return {"results": results}

@app.get("/images/{user_id}/{photo_id}")
async def get_image(user_id: str, photo_id: str):
    return FileResponse(os.path.join(BASE_IMAGE_FOLDER, user_id, f"{photo_id}.jpg"))

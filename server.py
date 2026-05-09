import json
from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
import torch
from PIL import Image
import io
from transformers import CLIPProcessor, CLIPModel
from fastapi.responses import FileResponse
import os
import requests
import base64
import concurrent.futures

MEMORY_FILE = "ai_long_term_memory.pt"
GOOGLE_API_KEY = "AIzaSyCVBozxMSHn7oNlYGc5nmqjVJ45rY8G3Uc"

app = FastAPI(title="ImageSearcher")

class SearchRequest(BaseModel):
    query: str

class AIBrain:
    def __init__(self):
        self.model_id = "openai/clip-vit-base-patch32"
        self.processor = CLIPProcessor.from_pretrained(self.model_id)
        self.model = CLIPModel.from_pretrained(self.model_id)
        
        if os.path.exists(MEMORY_FILE):
            self.memory = torch.load(MEMORY_FILE)
            print(f"🧠 Memory Restored: {len(self.memory)} users found.")
        else:
            self.memory = {} 
    
    def memorize_photo(self, image, photo_id, user_id):
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.get_image_features(pixel_values=inputs['pixel_values'])
            
            if hasattr(outputs, 'pooler_output'):
                features = outputs.pooler_output
            elif hasattr(outputs, 'image_embeds'):
                features = outputs.image_embeds
            elif not isinstance(outputs, torch.Tensor):
                features = outputs[0]
            else:
                features = outputs
                
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            
        if user_id not in self.memory:
            self.memory[user_id] = {}
            
        self.memory[user_id][photo_id] = features.detach()
        return len(self.memory[user_id])

    def search_memory(self, query, user_id):
        user_memory = self.memory.get(user_id, {})
        if not user_memory:
            return []
            
        text_inputs = self.processor(text=[query], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model.get_text_features(**text_inputs)
            
            if hasattr(outputs, 'pooler_output'):
                text_features = outputs.pooler_output
            elif hasattr(outputs, 'text_embeds'):
                text_features = outputs.text_embeds
            elif not isinstance(outputs, torch.Tensor):
                text_features = outputs[0]
            else:
                text_features = outputs
                
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

        matches = []
        for pid, photo_features in user_memory.items():
            score = torch.matmul(text_features, photo_features.T).item()
            matches.append((pid, score))
                
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

ai_brain = AIBrain()
BASE_IMAGE_FOLDER = "uploaded_images"

@app.post("/sync-photo")
async def sync_photo(user_id: str, file: UploadFile = File(...), photo_id: str = Form(...)):
    user_folder = os.path.join(BASE_IMAGE_FOLDER, user_id)
    os.makedirs(user_folder, exist_ok=True)
    
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image.thumbnail((800, 800))
    image.save(os.path.join(user_folder, f"{photo_id}.jpg"), optimize=True, quality=85)
    
    ai_brain.memorize_photo(image, photo_id, user_id)
    torch.save(ai_brain.memory, MEMORY_FILE)
    return {"status": "success"}

@app.get("/synced-photos")
async def get_synced_photos(user_id: str):
    user_memory = ai_brain.memory.get(user_id, {})
    return {"synced_ids": list(user_memory.keys())}

def analyze_single_image(best_id, score, query, user_id):
    file_path = os.path.join(BASE_IMAGE_FOLDER, user_id, f"{best_id}.jpg")
    with open(file_path, "rb") as img:
        base64_image = base64.b64encode(img.read()).decode("utf-8")

    prompt = f'''
    The user specifically searched for: "{query}". 
    Does this image strongly match the query? (Be highly strict about color matching and object identity).
    Reply in JSON with keys: 
    is_exact_match (bool),
    is_receipt (bool), 
    vendor (str), 
    total_price (str), 
    description (str).
    '''
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GOOGLE_API_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": base64_image}}]}]}
    
    try:
        api_response = requests.post(url, json=payload)
        response_data = api_response.json()
        
        if 'candidates' not in response_data:
            return None
            
        gemini_text = response_data['candidates'][0]['content']['parts'][0]['text']
        final_data = json.loads(gemini_text.replace('```json', '').replace('```', '').strip())
        
        if final_data.get('is_exact_match') is False:
            return None
            
        return {
            "image_found": f"{user_id}/{best_id}", 
            "image_base64": base64_image,
            "is_receipt": final_data.get('is_receipt', False),
            "vendor": final_data.get('vendor', 'Unknown'),
            "total_price": str(final_data.get('total_price', '0')),
            #"non_receipt_description": final_data.get('description', 'No description provided.'),
            "confidence": f"{score:.2f}"
        }
    except Exception as e:
        print(f"Error processing {best_id}: {e}")
        return None

@app.post("/search-and-extract")
async def search_and_extract(user_id: str, request: SearchRequest):
    matches = ai_brain.search_memory(request.query, user_id)
    
    if not matches:
        return {"error": "Your gallery has no photos synced yet!"}

    best_score = matches[0][1]
    close_matches = [m for m in matches if m[1] >= (best_score * 0.85)]
    top_n_matches = close_matches[:10]

    all_results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(analyze_single_image, best_id, score, request.query, user_id) for best_id, score in top_n_matches]
        
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result: 
                all_results.append(result)

    if not all_results:
        return {"error": f"No exact matches found for '{request.query}'. Try adjusting your search term."}

    all_results.sort(key=lambda x: float(x['confidence']), reverse=True)

    return {"results": all_results}

@app.get("/images/{user_id}/{photo_id}")
async def get_image(user_id: str, photo_id: str):
    return FileResponse(os.path.join(BASE_IMAGE_FOLDER, user_id, f"{photo_id}.jpg"))

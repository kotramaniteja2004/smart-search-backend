import os
import io
import json
import base64
import requests
import concurrent.futures
from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from PIL import Image
from supabase import create_client, Client

app = FastAPI(title="AI Smart Search - Supabase Edition")

# --- CONFIGURATION ---
# Replace these with your actual Supabase details
SUPABASE_URL = "https://wnzwkbcoxaxfulmyazwn.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InduendrYmNveGF4ZnVsbXlhenduIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3ODM0MTE0MCwiZXhwIjoyMDkzOTE3MTQwfQ.ak8-5FAOuY7a8tLr1E6f5aYg81jbBhS3Z0PiEVI84ks"
GOOGLE_API_KEY = "AIzaSyAVq0UMANz7ttMxspV587u6hGKQjE1ADe0".strip()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

class SearchRequest(BaseModel):
    query: str

@app.post("/sync-photo")
async def sync_photo(user_id: str, file: UploadFile = File(...), photo_id: str = Form(...)):
    try:
        image_bytes = await file.read()
        
        # 1. Upload Image to Supabase Storage
        file_path = f"{user_id}/{photo_id}.jpg"
        supabase.storage.from_("photos").upload(
            path=file_path,
            file=image_bytes,
            file_options={"content-type": "image/jpeg"}
        )
        
        # 2. Get Public URL for the image
        image_url = supabase.storage.from_("photos").get_public_url(file_path)

        # 3. Get Gemini Description
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        prompt = "Describe this image in 50 words for a search index. Include colors, objects, and text."
        gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:generateContent?key={GOOGLE_API_KEY}"
        payload = {"contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": base64_image}}]}]}
        
        resp = requests.post(gemini_url, json=payload).json()
        description = resp['candidates'][0]['content']['parts'][0]['text']

        # 4. Save Metadata to Supabase Database
        supabase.table("metadata").insert({
            "user_id": user_id,
            "photo_id": photo_id,
            "description": description
        }).execute()

        print(f"✅ Synced: {photo_id} for User: {user_id}")

        return {"status": "success", "url": image_url}
    except Exception as e:
        print(f"❌ SUPABASE/GEMINI ERROR: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.get("/synced-photos")
async def get_synced_photos(user_id: str):
    # Query the DB for all photo IDs belonging to this user
    response = supabase.table("metadata").select("photo_id").eq("user_id", user_id).execute()
    ids = [item['photo_id'] for item in response.data]
    return {"synced_ids": ids}

def verify_match(item, query):
    # item contains: photo_id, user_id, description
    photo_id = item['photo_id']
    user_id = item['user_id']
    description = item['description']
    
    # Get image from storage to show Gemini for verification
    file_path = f"{user_id}/{photo_id}.jpg"
    image_url = supabase.storage.from_("photos").get_public_url(file_path)
    
    # Simple keyword filter first
    if not any(word in description.lower() for word in query.lower().split()):
        return None

    # Ask Gemini if it's a match
    prompt = f'Search: "{query}". Desc: "{description}". Match? Reply ONLY JSON: {{"match": bool, "is_receipt": bool, "vendor": str, "total_price": str}}'
    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:generateContent?key={GOOGLE_API_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]} # Speeding it up by not sending image again
    
    try:
        resp = requests.post(gemini_url, json=payload).json()
        text = resp['candidates'][0]['content']['parts'][0]['text']
        data = json.loads(text.replace('```json', '').replace('```', '').strip())
        
        if not data.get('match'): return None
        
        return {
            "image_found": f"{user_id}/{photo_id}",
            "image_url": image_url, # Now sending a permanent URL!
            "is_receipt": data.get('is_receipt', False),
            "vendor": data.get('vendor', 'Unknown'),
            "total_price": str(data.get('total_price', '0'))
        }
    except: return None

@app.post("/search-and-extract")
async def search_and_extract(user_id: str, request: SearchRequest):
    # Get all descriptions for this user from Supabase
    response = supabase.table("metadata").select("*").eq("user_id", user_id).execute()
    user_photos = response.data
    
    if not user_photos:
        return {"error": "No photos synced yet!"}

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(verify_match, item, request.query) for item in user_photos]
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            if res: results.append(res)

    return {"results": results}

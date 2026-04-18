# Cell 1
from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
import torch
from PIL import Image
import io
from transformers import CLIPProcessor, CLIPModel
import google.generativeai as genai
from fastapi.responses import FileResponse
import os

# PLACE YOUR KEY HERE
genai.configure(api_key="AIzaSyDn0nuEYky3fdSZnHf097upDjpiqajZBwY")

app = FastAPI(title="AI SEARCH BRAIN APP")

class SearchRequest(BaseModel):
    search_query: str



# Cell 2
class AIBrain:
    def __init__(self):
        self.model_id = "openai/clip-vit-base-patch32"
        self.processor = CLIPProcessor.from_pretrained(self.model_id)
        self.model = CLIPModel.from_pretrained(self.model_id)
        self.memory = {}

    def memorize_photo(self, image, photo_id):
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            # Force it to look ONLY at the pixels
            features = self.model.get_image_features(pixel_values=inputs['pixel_values'])
            
            # THE CROWBAR: If it's still a box, break it open
            if hasattr(features, 'pooler_output'):
                features = features.pooler_output
            elif type(features) is tuple or type(features) is list:
                features = features[0]
                
            # Now we can safely do the math on the raw numbers
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            
        self.memory[photo_id] = features.detach()
        return len(self.memory)

    def search_memory(self, query):
        if not self.memory:
            return None, -1.0
            
        text_inputs = self.processor(text=[query], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            # Force it to look ONLY at the text
            text_features = self.model.get_text_features(
                input_ids=text_inputs['input_ids'], 
                attention_mask=text_inputs['attention_mask']
            )
            
            # THE CROWBAR: Unbox the text features
            if hasattr(text_features, 'pooler_output'):
                text_features = text_features.pooler_output
            elif type(text_features) is tuple or type(text_features) is list:
                text_features = text_features[0]

            # Do the math
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

        best_id = None
        highest_score = -1.0

        for pid, photo_features in self.memory.items():
            # Calculate the final similarity score
            score = torch.matmul(text_features, photo_features.T).item()
            if score > highest_score:
                highest_score = score
                best_id = pid
                
        return best_id, highest_score

# Initialize one instance of the brain
ai_brain = AIBrain()





# Cell 3


# Create a folder on your Mac to hold the physical images
os.makedirs("uploaded_images", exist_ok=True)

@app.post("/sync-photo")
async def sync_photo(file: UploadFile = File(...), photo_id: str = Form(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # --- NEW: Save the actual picture to the Mac's hard drive ---
        image.save(f"uploaded_images/{photo_id}.jpg")
        
        # Save the math to our Brain object
        total_in_memory = ai_brain.memorize_photo(image, photo_id)
        
        print(f"🧠 Memorized Photo ID: {photo_id} | Total in Brain: {total_in_memory}")
        return {"status": "success"}
    except Exception as e:
        print(f"❌ Sync Error: {e}")
        return {"status": "error", "detail": str(e)}

# Update just this section in Cell 3
@app.post("/search-and-extract")
async def search_and_extract(request: SearchRequest):
    try:
        best_id, score = ai_brain.search_memory(request.search_query)
        
        # --- THE ACCURACY FIX ---
        # Set a minimum confidence score (0.26 is a good starting point)
        THRESHOLD = 0.26 
        
        if best_id is None:
            return {"error": "No photos in memory"}
            
        if score < THRESHOLD:
            print(f"⚠️ Low confidence ({score:.4f}). Ignoring wrong match.")
            return {"error": "No matching photo found in your synced gallery."}
        # -------------------------

        print(f"🎯 AI Match Found! ID: {best_id} (Score: {score:.4f})")
        return {
            "image_found": str(best_id),
            "vendor": "Local AI Search",
            "total_price": 0.0,
            "confidence": f"{score:.2f}"
        }
    except Exception as e:
        print(f"❌ Search Error: {e}")
        return {"error": str(e)}

# --- NEW: The Endpoint to serve the image back to the phone ---
@app.get("/images/{photo_id}")
async def get_image(photo_id: str):
    # The phone will hit this URL, and we send the file back
    file_path = f"uploaded_images/{photo_id}.jpg"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return {"error": "Image file not found on server"}
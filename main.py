from fastapi import FastAPI, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from groq import AsyncGroq
from typing import Optional
import os
import json

app = FastAPI()

# CORS - allows your frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Groq client
client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))

# Store chat history per user session
conversation_histories = {}

# AI personalities
MODES = {
    "tutor": """You are TutorAI, a world-class AI tutor.
- Explain things simply with real examples
- Ask follow-up questions to check understanding
- Be warm, encouraging, patient
- Format with clear structure""",
    
    "code": """You are CodeAI, an expert programmer.
- Write clean, well-commented code
- Explain WHY code works, not just what it does
- Always use proper code blocks with language specified
- Spot and fix bugs clearly""",
    
    "think": """You are ThinkAI, a deep analytical reasoner.
- Think step by step through complex problems
- Show your reasoning process
- Be honest about uncertainty
- Explore multiple perspectives""",
    
    "creative": """You are CreativeAI, an imaginative creative partner.
- Generate unique, original ideas
- Build on the user's ideas and make them better
- Write with style, flair, and personality
- Always give multiple options/variations""",
}

class Message(BaseModel):
    message: str
    mode: str = "tutor"
    pdf_context: str = ""
    image_data: Optional[str] = None

@app.get("/")
def home():
    return {"status": "TutorAI backend is live!"}

@app.post("/chat")
async def chat(data: Message, session_id: str = Header(default="default")):
    system = MODES.get(data.mode, MODES["tutor"])
    
    # Add PDF context if provided
    if data.pdf_context:
        system = f"You have access to this document:\n---\n{data.pdf_context}\n---\n" + system

    # Get or create history for this session
    history = conversation_histories.setdefault(session_id, [])

    # Handle image upload
    if data.image_data:
        try:
            header, base64_str = data.image_data.split(",", 1)
            media_type = header.split(":")[1].split(";")[0]
        except Exception:
            media_type = "image/jpeg"
            base64_str = data.image_data

        # Only text messages for history (images can't go in history for text models)
        text_only_history = [m for m in history if isinstance(m.get("content"), str)]
        
        messages_to_send = (
            [{"role": "system", "content": system}]
            + text_only_history
            + [{"role": "user", "content": [
                {"type": "text", "text": data.message or "Describe and analyze this image in detail."},
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{base64_str}"}}
            ]}]
        )
        model = "meta-llama/llama-4-scout-17b-16e-instruct"
    else:
        # Normal text chat
        history.append({"role": "user", "content": data.message})
        messages_to_send = [{"role": "system", "content": system}] + history
        model = "llama-3.3-70b-versatile"

    async def generate():
        full_reply = ""
        try:
            # Stream from Groq
            stream = await client.chat.completions.create(
                model=model,
                messages=messages_to_send,
                max_completion_tokens=1024,
                temperature=0.7,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    full_reply += delta
                    yield f"data: {json.dumps({'chunk': delta})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return

        # Save to history
        if data.image_data:
            history.append({"role": "user", "content": data.message or "[User shared an image]"})
        history.append({"role": "assistant", "content": full_reply})
        
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.post("/reset")
async def reset(session_id: str = Header(default="default")):
    conversation_histories.pop(session_id, None)
    return {"status": "reset"}

# For local testing only
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

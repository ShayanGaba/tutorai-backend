from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from groq import Groq
from typing import Optional
import os

app = FastAPI()

# ✅ USE THIS instead of custom middleware - FastAPI's official CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
conversation_history = []

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
def chat(data: Message):
    system = MODES.get(data.mode, MODES["tutor"])

    if data.pdf_context:
        system = f"You have access to this document:\n---\n{data.pdf_context}\n---\n" + system

    if data.image_data:
        try:
            header, base64_str = data.image_data.split(",", 1)
            media_type = header.split(":")[1].split(";")[0]
        except Exception:
            media_type = "image/jpeg"
            base64_str = data.image_data

        user_message_content = [
            {
                "type": "text",
                "text": data.message if data.message else "What do you see in this image? Describe and analyze it in detail."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{base64_str}"
                }
            }
        ]
    else:
        user_message_content = data.message

    conversation_history.append({
        "role": "user",
        "content": user_message_content
    })

    # ✅ vision model for images, fast model for text
    model = "meta-llama/llama-4-scout-17b-16e-instruct" if data.image_data else "llama-3.3-70b-versatile"

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}] + conversation_history,
        max_tokens=1024,
        temperature=0.7,
    )

    reply = response.choices[0].message.content

    conversation_history.append({
        "role": "assistant",
        "content": reply
    })

    return {"reply": reply}

@app.post("/reset")
def reset():
    conversation_history.clear()
    return {"status": "reset"}

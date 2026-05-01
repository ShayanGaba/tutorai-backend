from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from groq import Groq
from typing import Optional
import os
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

        text_only_history = [
            m for m in conversation_history
            if isinstance(m["content"], str)
        ]

        messages_to_send = (
            [{"role": "system", "content": system}]
            + text_only_history
            + [{"role": "user", "content": [
                {"type": "text", "text": data.message if data.message else "Describe and analyze this image in detail."},
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{base64_str}"}}
            ]}]
        )
        model = "meta-llama/llama-4-scout-17b-16e-instruct"
    else:
        conversation_history.append({"role": "user", "content": data.message})
        messages_to_send = [{"role": "system", "content": system}] + conversation_history
        model = "llama-3.3-70b-versatile"

    def generate():
        full_reply = ""
        with client.chat.completions.stream(
            model=model,
            messages=messages_to_send,
            max_tokens=1024,
            temperature=0.7,
        ) as stream:
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    full_reply += delta
                    yield f"data: {json.dumps({'chunk': delta})}\n\n"

        if data.image_data:
            conversation_history.append({"role": "user", "content": data.message if data.message else "[User shared an image]"})
        conversation_history.append({"role": "assistant", "content": full_reply})
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.post("/reset")
def reset():
    conversation_history.clear()
    return {"status": "reset"}

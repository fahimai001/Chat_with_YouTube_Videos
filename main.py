from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, HttpUrl
import uvicorn
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from src.helper_func import extract_transcript, build_qa_chain, ask_question

app = FastAPI(
    title="YouTube Transcript QA System",
    description="Web interface for YouTube transcript extraction and Q&A",
    version="1.0.0"
)

# Setup templates and static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class TranscriptRequest(BaseModel):
    url: HttpUrl
    language: str = "en"

class QuestionRequest(BaseModel):
    url: HttpUrl
    question: str
    language: str = "en"

qa_chains = {}

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/transcript")
async def get_transcript(request: TranscriptRequest):
    """Get transcript endpoint"""
    transcript = extract_transcript(str(request.url), request.language)
    if transcript.startswith("Error:"):
        raise HTTPException(status_code=400, detail=transcript)
    return {"transcript": transcript}

@app.post("/ask")
async def ask_video_question(request: QuestionRequest):
    """Q&A endpoint"""
    url_str = str(request.url)
    
    # Get or create QA chain
    if url_str not in qa_chains:
        transcript = extract_transcript(url_str, request.language)
        if transcript.startswith("Error:"):
            raise HTTPException(status_code=400, detail=transcript)
        qa_chains[url_str] = build_qa_chain(transcript)
    
    # Get answer
    answer = ask_question(qa_chains[url_str], request.question)
    if answer.startswith("Error:"):
        raise HTTPException(status_code=500, detail=answer)
    
    return {"answer": answer}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
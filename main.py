from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, HttpUrl
from typing import Optional
import uvicorn

from src.helper_func import extract_transcript, build_qa_chain, ask_question

app = FastAPI(
    title="YouTube Transcript QA API",
    description="API for extracting YouTube transcripts and answering questions about video content",
    version="1.0.0"
)

class TranscriptRequest(BaseModel):
    url: HttpUrl
    language: str = "en"

class TranscriptResponse(BaseModel):
    transcript: str

class QuestionRequest(BaseModel):
    url: HttpUrl
    question: str
    language: str = "en"

class QuestionResponse(BaseModel):
    answer: str

qa_chains = {}

@app.post("/transcript", response_model=TranscriptResponse)
async def get_transcript(request: TranscriptRequest):
    """Extract transcript from a YouTube video URL"""
    transcript = extract_transcript(str(request.url), request.language)
    
    if transcript.startswith("Error:"):
        raise HTTPException(status_code=400, detail=transcript)
    
    return TranscriptResponse(transcript=transcript)

@app.post("/ask", response_model=QuestionResponse)
async def ask_video_question(request: QuestionRequest):
    """Ask a question about a YouTube video"""
    url_str = str(request.url)
    

    transcript = extract_transcript(url_str, request.language)
    if transcript.startswith("Error:"):
        raise HTTPException(status_code=400, detail=transcript)
    

    if url_str not in qa_chains:
        qa_chains[url_str] = build_qa_chain(transcript)
    

    answer = ask_question(qa_chains[url_str], request.question)
    if answer.startswith("Error:"):
        raise HTTPException(status_code=500, detail=answer)
    
    return QuestionResponse(answer=answer)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
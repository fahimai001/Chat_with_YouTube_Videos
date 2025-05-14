# YouTube Transcript QA System

A multilingual web application that allows users to ask questions about YouTube videos and get answers based on the video's transcript.

## Features

- Extract transcripts from YouTube videos
- Ask questions about video content in multiple languages
- Retrieval-based Q&A using AI embeddings
- Web interface for easy interaction

## Requirements

- Python 3.8+
- Google API key for Gemini model
- YouTube Transcript API

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your Google API key:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

## Usage

Run the application:
```
uvicorn main:app --reload
```

The web interface will be available at http://localhost:8000

## API Endpoints

- `GET /`: Main web interface
- `POST /transcript`: Get transcript from YouTube URL
- `POST /ask`: Ask questions about a video

## Technologies

- FastAPI
- LangChain
- Google Gemini AI
- FAISS Vector Store
- YouTube Transcript API
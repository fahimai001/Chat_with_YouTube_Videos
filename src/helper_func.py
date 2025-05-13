import os
import re
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=API_KEY)

def extract_transcript(url: str, lang: str = "en") -> str:
    """Extract and return transcript text from a YouTube video URL."""
    video_id_match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    if not video_id_match:
        return "Error: Invalid YouTube URL"
    
    video_id = video_id_match.group(1)
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
    except Exception as e:
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            first_transcript = next(iter(transcript_list))
            if first_transcript.language_code != lang:
                msg = f"Transcript not available in {lang}. Using {first_transcript.language_code}."
                print(msg)
            transcript = first_transcript.fetch()
        except Exception as e:
            return f"Error: Could not retrieve transcript. {str(e)}"
    

    texts = []
    for entry in transcript:
        try:
            texts.append(entry['text'])  
        except TypeError:
            texts.append(entry.text)    
    return ' '.join(texts)

def build_qa_chain(transcript: str):
    """Build a retrieval-based QA chain using transcript."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(transcript)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_KEY)
    vectorstore = FAISS.from_texts(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    template = """
    You are a multilingual AI assistant that answers questions about YouTube videos. 
    Always respond in the same language as the question. 
    If the transcript language differs from the question language, translate the answer while preserving meaning.

    Context from video transcript:
    {context}

    Question: {question}

    Provide a concise answer based only on the transcript. If no relevant information exists, say:
    "The transcript doesn't contain relevant information for this question." 
    Maintain the question's language in your response.
    """
    prompt = ChatPromptTemplate.from_template(template)

    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

def ask_question(chain, question: str) -> str:
    """Ask a question using the given retrieval chain."""
    try:
        return chain.invoke(question)
    except Exception as e:
        return f"Error processing question: {str(e)}"
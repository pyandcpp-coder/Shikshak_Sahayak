import os
import uuid
import asyncio
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import re

# Load environment variables first
load_dotenv()

app = FastAPI(title="Shiksha Sahayak Backend", version="1.0.0")

# CORS configuration
origins = ["*"] 

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize directories
os.makedirs("temp_files", exist_ok=True)
os.makedirs("vector_stores", exist_ok=True)

# Global variables for lazy loading
llm = None
embedding_model = None
models_initialized = False

def initialize_models():
    """Initialize models lazily to speed up startup"""
    global llm, embedding_model, models_initialized
    
    if models_initialized:
        return True
        
    try:
        # Import heavy dependencies only when needed
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_community.embeddings import HuggingFaceEmbeddings
        
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0.3, 
            convert_system_message_to_human=True,
            google_api_key=google_api_key
        )
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        models_initialized = True
        print("Models initialized successfully")
        return True
    except Exception as e:
        print(f"Error initializing models: {e}")
        return False

# Request models
class UrlRequest(BaseModel):
    url: str

class ChatRequest(BaseModel):
    session_id: str
    query: str

class QuizRequest(BaseModel):
    session_id: str

@app.get("/") 
def read_root():
    return {"status": "Shiksha Sahayak Backend is running!", "version": "1.0.0"}

@app.get("/health")
def health_check():
    """Health check endpoint for Cloud Run"""
    return {
        "status": "healthy", 
        "models_loaded": models_initialized,
        "port": os.environ.get("PORT", "8080")
    }

@app.post("/process-pdf")
async def process_pdf_endpoint(file: UploadFile = File(...)):
    """
    Receives a PDF file, processes it through the RAG pipeline,
    and returns a unique session ID for future interactions.
    """
    if not initialize_models():
        return {"status": "error", "message": "Failed to initialize models"}
    
    # Import heavy dependencies only when needed
    from rag_processor import load_and_split_pdf, create_vector_store
    
    temp_dir = "temp_files"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.filename)

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    session_id = str(uuid.uuid4())

    try:
        chunks = load_and_split_pdf(file_path)
        create_vector_store(chunks, session_id)
        status = "success"
        message = "PDF processed successfully."
    except Exception as e:
        status = "error"
        message = f"An error occurred: {str(e)}"
        session_id = None

    # Clean up temp file
    try:
        os.remove(file_path)
    except:
        pass

    return {"status": status, "message": message, "session_id": session_id}

@app.post("/chat")
def chat_with_doc(request: ChatRequest):
    """
    Handles chat queries for a specific document session.
    """
    if not initialize_models():
        return {"status": "error", "message": "Failed to initialize models"}
    
    # Import heavy dependencies only when needed
    from langchain.chains import RetrievalQA
    from langchain_community.vectorstores import Chroma
    
    session_path = os.path.join("vector_stores", request.session_id)
    if not os.path.exists(session_path):
        return {"status": "error", "message": "Session not found. Please upload a document first."}

    try:
        vector_store = Chroma(persist_directory=session_path, embedding_function=embedding_model)
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever()
        )
        
        response = qa_chain.run(request.query)
        return {"status": "success", "response": response}
    except Exception as e:
        return {"status": "error", "message": f"An error occurred: {str(e)}"}

@app.post("/generate-quiz")
def generate_quiz(request: QuizRequest):
    if not initialize_models():
        return {"status": "error", "message": "Failed to initialize models"}
    
    # Import heavy dependencies only when needed
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain, RetrievalQA
    from langchain.chains.combine_documents.stuff import StuffDocumentsChain
    from langchain_community.vectorstores import Chroma

    session_path = os.path.join("vector_stores", request.session_id)
    if not os.path.exists(session_path):
        return {"status": "error", "message": "Session not found."}

    try:
        vector_store = Chroma(persist_directory=session_path, embedding_function=embedding_model)

        prompt_template = PromptTemplate(
            input_variables=["context"],
            template="""
            Based on the content of the document, generate a JSON array of 5 multiple-choice questions.

            Each object in the array must have:
            - "question": A string
            - "options": A list of 4 strings
            - "correct_answer": One of the strings from the "options" list

            ⚠️ Return only the raw JSON array without any explanation or formatting like triple backticks.

            Context: {context}
            """
        )

        llm_chain = LLMChain(llm=llm, prompt=prompt_template)
        stuff_chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_variable_name="context"
        )

        qa_chain = RetrievalQA(
            retriever=vector_store.as_retriever(),
            combine_documents_chain=stuff_chain
        )

        response = qa_chain.run("Generate a quiz from the document.")
        cleaned_response = re.sub(r"```(?:json)?", "", response)
        cleaned_response = cleaned_response.replace("```", "").strip()

        return {"status": "success", "quiz_data": cleaned_response}
    except Exception as e:
        return {"status": "error", "message": f"An error occurred: {str(e)}"}

@app.post("/process-url")
async def process_url_endpoint(request: UrlRequest):
    """
    Receives a URL, scrapes it, processes it through the RAG pipeline,
    and returns a unique session ID.
    """
    if not initialize_models():
        return {"status": "error", "message": "Failed to initialize models"}
    
    # Import heavy dependencies only when needed
    from rag_processor import load_and_split_url, create_vector_store
    
    session_id = str(uuid.uuid4())
    try:
        chunks = load_and_split_url(request.url)
        create_vector_store(chunks, session_id)
        status = "success"
        message = "URL processed successfully."
    except Exception as e:
        status = "error"
        message = f"An error occurred processing the URL: {str(e)}"
        session_id = None
    
    return {"status": status, "message": message, "session_id": session_id}

@app.post("/process-youtube")
async def process_youtube_endpoint(request: UrlRequest):
    """
    Receives a YouTube URL, transcribes the audio, processes it,
    and returns a unique session ID.
    """
    if not initialize_models():
        return {"status": "error", "message": "Failed to initialize models"}
    
    # Import heavy dependencies only when needed
    from rag_processor import load_and_split_youtube_url, create_vector_store
    
    session_id = str(uuid.uuid4())
    try:
        chunks = load_and_split_youtube_url(request.url)
        create_vector_store(chunks, session_id)
        status = "success"
        message = "YouTube video processed successfully."
    except Exception as e:
        status = "error"
        message = f"An error occurred processing the YouTube video: {str(e)}"
        session_id = None
    
    return {"status": status, "message": message, "session_id": session_id}

@app.post("/generate-summary")
def generate_summary(request: QuizRequest): 
    """
    Generates a concise summary of the document content for a specific session.
    """
    if not initialize_models():
        return {"status": "error", "message": "Failed to initialize models"}
    
    # Import heavy dependencies only when needed
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    from langchain_community.vectorstores import Chroma
    
    session_path = os.path.join("vector_stores", request.session_id)
    if not os.path.exists(session_path):
        return {"status": "error", "message": "Session not found."}

    try:
        vector_store = Chroma(persist_directory=session_path, embedding_function=embedding_model)
        
        prompt_template = """
        Write a concise summary of the following text, highlighting the main ideas and key takeaways. 
        Present the summary as a few clear bullet points.

        Context: {context}

        Summary:
        """

        summary_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
            chain_type_kwargs={
                "prompt": PromptTemplate(
                    template=prompt_template,
                    input_variables=["context"]
                )
            }
        )

        response = summary_chain.run("Generate a summary of the document.")
        return {"status": "success", "summary": response}
    except Exception as e:
        return {"status": "error", "message": f"An error occurred: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
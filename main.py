import os
import uuid
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from rag_processor import load_and_split_pdf, create_vector_store,load_and_split_url,load_and_split_youtube_url
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

app = FastAPI()
import re





# --- The rest of your main.py file remains the same ---
# ... (imports, origins, app.add_middleware, etc.) ...
# --- CORS Configuration ---
# This is crucial for allowing the frontend (running on a different URL)
# to communicate with this backend.
origins = ["*"] # In production, you'd restrict this to your frontend's domain

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class UrlRequest(BaseModel):
    url: str
@app.get("/") 
def read_root():
    return {"status": "Shiksha Sahayak Backend is running!"}

# --- PDF Processing Endpoint ---
@app.post("/process-pdf")
async def process_pdf_endpoint(file: UploadFile = File(...)):
    """
    Receives a PDF file, processes it through the RAG pipeline,
    and returns a unique session ID for future interactions.
    """
    # 1. Save the uploaded file temporarily
    temp_dir = "temp_files"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.filename)

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    # 2. Generate a unique session ID
    session_id = str(uuid.uuid4())

    # 3. Process the PDF using our rag_processor
    try:
        chunks = load_and_split_pdf(file_path)
        create_vector_store(chunks, session_id)
        status = "success"
        message = "PDF processed successfully."
    except Exception as e:
        status = "error"
        message = f"An error occurred: {str(e)}"
        session_id = None

    os.remove(file_path)

    return {"status": status, "message": message, "session_id": session_id}

# main.py (add this new section of code)
# You'll need to add your Google API Key here

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os

load_dotenv()


google_api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, convert_system_message_to_human=True)

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


class ChatRequest(BaseModel):
    session_id: str
    query: str

class QuizRequest(BaseModel):
    session_id: str

@app.post("/chat")
def chat_with_doc(request: ChatRequest):
    """
    Handles chat queries for a specific document session.
    """
    session_path = os.path.join("vector_stores", request.session_id)
    if not os.path.exists(session_path):
        return {"status": "error", "message": "Session not found. Please upload a document first."}

    vector_store = Chroma(persist_directory=session_path, embedding_function=embedding_model)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )
    try:
        response = qa_chain.run(request.query)
        return {"status": "success", "response": response}
    except Exception as e:
        return {"status": "error", "message": f"An error occurred: {str(e)}"}


@app.post("/generate-quiz")
def generate_quiz(request: QuizRequest):
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain.chains.combine_documents.stuff import StuffDocumentsChain

    session_path = os.path.join("vector_stores", request.session_id)
    if not os.path.exists(session_path):
        return {"status": "error", "message": "Session not found."}

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
    import re 

    try:
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
    session_path = os.path.join("vector_stores", request.session_id)
    if not os.path.exists(session_path):
        return {"status": "error", "message": "Session not found."}

    vector_store = Chroma(persist_directory=session_path, embedding_function=embedding_model)
    
    prompt_template = """
    Write a concise summary of the following text, highlighting the main ideas and key takeaways. 
    Present the summary as a few clear bullet points.

    Context: {context}

    Summary:
    """
    from langchain.prompts import PromptTemplate

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

    try:
        response = summary_chain.run("Generate a summary of the document.")
        return {"status": "success", "summary": response}
    except Exception as e:
        return {"status": "error", "message": f"An error occurred: {str(e)}"}

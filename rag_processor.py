import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from newspaper import Article
import whisper
from pytube import YouTube
print("All libraries imported successfully.")

# rag_processor.py (add this below the imports)

def load_and_split_pdf(pdf_path: str):
    """
    Loads a PDF from the given path and splits it into smaller text chunks.

    Args:
        pdf_path: The file path to the PDF document.

    Returns:
        A list of 'Document' objects, where each object is a chunk of text.
    """
    print(f"Loading and splitting PDF from: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()


    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    print(f"Successfully split the document into {len(chunks)} chunks.")
    return chunks

# rag_processor.py (add this below the previous function)

def create_vector_store(chunks, session_id: str):
    """
    Creates a ChromaDB vector store from the text chunks and saves it to disk.

    Args:
        chunks: A list of text chunks (Document objects).
        session_id: A unique identifier for the current user session.
    
    Returns:
        The path to the persistent vector store directory.
    """
    # 1. Define the embedding model
    # This model will convert our text chunks into numerical vectors.
    # "all-MiniLM-L6-v2" is a popular, fast, and effective model.
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 2. Define the path for the persistent vector store
    # We use the session_id to create a unique folder for each uploaded PDF.
    persist_directory = os.path.join("vector_stores", session_id)
    print(f"Creating vector store at: {persist_directory}")

    # 3. Create and persist the vector store
    # This is the core step. ChromaDB takes the chunks, uses the embedding model
    # to convert them to vectors, and saves everything to the specified directory.
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )

    
    print("Vector store created successfully.")
    return persist_directory

# rag_processor.py (add this new function)

from newspaper import Article

def load_and_split_url(url: str):
    """
    Loads content from a URL, extracts the main article, and splits it into chunks.
    """
    print(f"Loading and splitting content from URL: {url}")

    article = Article(url)
    article.download()
    article.parse()
    
    # 2. We use the article's text as our document content
    # We wrap it in a list of one to match the structure from PyPDFLoader
    # The page_content is the main text, and metadata helps us know the source.
    documents = [
        {"page_content": article.text, "metadata": {"source": url}}
    ]

    # 3. Split the document into chunks (reusing our existing splitter)
    # Note: We need to adapt this slightly as we don't have LangChain 'Document' objects yet.
    # Let's refine this to be more robust.

    from langchain.docstore.document import Document as LangchainDocument

    # Convert our parsed data into the format LangChain expects
    langchain_docs = [LangchainDocument(page_content=article.text, metadata={"source": url})]
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(langchain_docs)

    print(f"Successfully split the URL content into {len(chunks)} chunks.")
    return chunks


import os
import shutil
import subprocess
import whisper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument

def load_and_split_youtube_url(url: str):
    """
    Downloads audio from a YouTube URL using yt-dlp, transcribes it with Whisper, 
    and splits the text into chunks for further processing.
    """
    print(f"Processing YouTube URL: {url}")

    print("Downloading audio with yt-dlp...")
    temp_dir = "temp_audio"
    os.makedirs(temp_dir, exist_ok=True)
    
    command = [
        'yt-dlp',
        '-x',
        '--audio-format', 'mp3',
        '-o', f'{temp_dir}/%(title)s.%(ext)s',
        url
    ]
    
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"yt-dlp failed: {e}")

    audio_files = [f for f in os.listdir(temp_dir) if f.endswith('.mp3')]
    if not audio_files:
        raise FileNotFoundError("No MP3 file found after yt-dlp download.")
    
    audio_file_path = os.path.join(temp_dir, audio_files[0])
    print(f"Audio downloaded to: {audio_file_path}")

    print("Transcribing audio... (This may take a while for long videos)")
    whisper_model = whisper.load_model("base")
    transcription_result = whisper_model.transcribe(audio_file_path, fp16=False)
    transcribed_text = transcription_result['text']
    print("Transcription complete.")

    shutil.rmtree(temp_dir)

    langchain_docs = [LangchainDocument(page_content=transcribed_text, metadata={"source": url})]
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(langchain_docs)

    print(f"Successfully split the YouTube transcription into {len(chunks)} chunks.")
    return chunks

if __name__ == '__main__':
    TEST_SESSION_ID = "test_session_123"

    # 2. Define the path to our sample PDF
    pdf_path = os.path.join("test_docs", "test.pdf")

    # 3. Check if the sample PDF exists
    if not os.path.exists(pdf_path):
        print(f"Error: Test PDF not found at {pdf_path}")
        print("Please create a 'test_docs' folder and place a 'sample.pdf' file inside it.")
    else:
        # 4. Run the entire pipeline
        chunks = load_and_split_pdf(pdf_path)
        if chunks:
            vector_store_path = create_vector_store(chunks, TEST_SESSION_ID)
            print(f"--- RAG Pipeline Test Complete ---")
            print(f"Vector store for session '{TEST_SESSION_ID}' is ready at: {vector_store_path}")
# Shiksha Sahayak - Backend

This is the backend service for **Shiksha Sahayak**, an AI-powered learning assistant. Built with **FastAPI**, this server handles the core logic of the application, including content processing, vectorization, and interaction with a Large Language Model (LLM) via LangChain.

The backend exposes a set of RESTful API endpoints that the [Shiksha Sahayak Frontend](https://github.com/pyandcpp-coder/Shikshak_Sahayak_Frontend?tab=readme-ov-file) consumes to provide its features.

---

## ✨ Key Features & Technologies

-   **High-Performance API**: Built with [**FastAPI**](https://fastapi.tiangolo.com/) for modern, fast, and asynchronous request handling.
-   **RAG Pipeline**: Implements a full Retrieval-Augmented Generation pipeline.
    -   **Content Ingestion**: Processes PDFs, website URLs, and YouTube video transcripts.
    -   **Text Splitting**: Uses `RecursiveCharacterTextSplitter` from LangChain to chunk documents.
    -   **Vectorization**: Employs `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`) to create text embeddings.
    -   **Vector Storage**: Utilizes [**ChromaDB**](https://www.trychroma.com/) for efficient similarity search and retrieval.
-   **LLM Integration**: Leverages the [**Google Gemini**](https://ai.google.dev/models/gemini) model via `langchain-google-genai` for:
    -   Answering contextual questions about the documents.
    -   Generating summaries.
    -   Creating multiple-choice quizzes.
-   **Lazy Loading**: Models are initialized on the first request to ensure fast server startup times, which is ideal for serverless environments.
-   **Production-Ready**: Configured to run with `gunicorn` and `uvicorn` workers for stable, multi-process performance.

---

## API Endpoints

The following are the main endpoints provided by the service:

| Method | Endpoint                    | Description                                                                 |
| :----- | :-------------------------- | :-------------------------------------------------------------------------- |
| `GET`  | `/`                         | Root endpoint to check if the service is running.                           |
| `POST` | `/process-pdf`              | Upload a PDF file. Processes it and returns a `session_id`.                 |
| `POST` | `/process-url`              | Submit a website URL. Scrapes it and returns a `session_id`.                |
| `POST` | `/process-youtube`          | Submit a YouTube URL. Transcribes it and returns a `session_id`.            |
| `POST` | `/chat`                     | Ask a question related to a document using a valid `session_id`.            |
| `POST` | `/generate-summary`         | Generate a concise summary for the content of a given `session_id`.         |
| `POST` | `/generate-quiz`            | Generate a multiple-choice quiz for the content of a given `session_id`.    |

---

## 🚀 Getting Started

Follow these instructions to set up and run the backend server locally.

### Prerequisites

-   Python 3.9+
-   `pip` and `venv`
-   A [Google AI Studio API Key](https://aistudio.google.com/app/apikey)

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/pyandcpp-coder/shiksha-sahayak-backend.git
    cd shiksha-sahayak-backend
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    Create a file named `.env` in the root of the project and add your Google API key.

    ```env
    # .env
    GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY_HERE"
    ```

5.  **Run the development server:**
    ```bash
    uvicorn main:app --reload
    ```
    The server will be available at `http://127.0.0.1:8000`. You can access the interactive API documentation (Swagger UI) at `http://127.0.0.1:8000/docs`.

---

## 🌐 Deployment

This application is configured for deployment on cloud platforms like **Render**.

1.  **Push to GitHub:** Ensure your repository, including the `requirements.txt` file, is pushed to GitHub.
2.  **Create a Web Service on Render:**
    -   Connect your GitHub repository.
    -   **Build Command**: `pip install -r requirements.txt`
    -   **Start Command**: `gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app`
3.  **Configure Environment Variable:**
    -   In the Render service settings, go to the "Environment" tab.
    -   Add a new Secret File or Environment Variable with the key `GOOGLE_API_KEY` and paste your API key as the value.

Render will automatically build and deploy the service.

---

## 🔗 Frontend Repository

This backend is designed to be used with its corresponding frontend.

➡️ **Find the frontend repository here: [pyandcpp-coder/shiksha-sahayak-frontend](https://github.com/pyandcpp-coder/Shikshak_Sahayak_Frontend?tab=readme-ov-file)**
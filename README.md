# PDF RAG Agent Setup Instructions

Follow these steps to get the PDF RAG Agent up and running:

## Step 1: Install Requirements
1. install ollama and search for llama 3.2 and nomic embedding in ollama and install both 
  
2. Install Qdrant
   Install from [qdrant.tech](https://qdrant.tech/documentation/quickstart/) [first 2 cmds]
   u need docker desktop installed

## Step 2: Set Up Project

1. Create a new folder for your project
   ```bash
   mkdir pdf-rag-agent
   cd pdf-rag-agent
   ```

2. Create a virtual environment
   ```bash
   python -m venv .venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with:
   ```
   QDRANT_URL_LOCALHOST=http://localhost:6333
   ```

5. Create a folder for your PDFs
   ```bash
   mkdir pdf_documents
   ```

6. Copy your PDF files into the `pdf_documents` folder

## Step 3: Set Up the Agent

1. Create a file named `app.py` with the code from the provided PDF RAG Agent

2. Pull the required Ollama models
   ```bash
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```

3. Uncomment the knowledge base loading line in app.py (only for first run):
   ```python
   knowledge_base.load(upsert=True)
   ```

## Step 4: Run the Agent

1. Start Qdrant if not already running (in docker it will run)

2. Make sure Ollama is running

3. Run the application
   ```bash
   python app.py
   ```

4. Access the web interface
   - Open a browser and go to: http://localhost:8000

5. After first run, comment out the knowledge base loading line:
   ```python
   # knowledge_base.load(upsert=True)
   ```

## Step 5: Using the Agent

1. Ask questions about your PDF documents in the chat interface

2. The agent will search the knowledge base and respond with information from your PDFs

3. If you add new PDFs, uncomment and run the knowledge base loading line again

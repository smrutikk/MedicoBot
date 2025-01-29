from flask import Flask, request, jsonify
from flask_cors import CORS 
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.schema import Document
import os
from PyPDF2 import PdfReader

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})  

# Configure model and paths
MODEL_PATH = "./BioMistral-7B.Q4_K_M.gguf"  # Ensure this points to your local LlamaCpp model file
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "your_hf_api_key_here")

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

# Verify that the PDFs contain extractable text
pdf_paths = [
    "./pdfs/1.pdf",
    "./pdfs/2.pdf",
    "./pdfs/3.pdf"
]

documents = []

# Extract text from each PDF and wrap it in Document objects
for path in pdf_paths:
    text = extract_text_from_pdf(path)
    #print(f"Text from {path[:20]}...: {text[:500]}")  # Print the first 500 characters of each PDF text
    if text.strip():  # Only create Document if text is not empty
        documents.append(Document(page_content=text, metadata={"source": path}))
    else:
        print(f"Warning: No extractable text in {path}")

# Split the documents into chunks
try:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    if not chunks:
        raise ValueError("Document chunks are empty. Check the text splitting logic.")
except Exception as e:
    print(f"Error during document splitting: {e}")
    raise

# Create embeddings and vector store
try:
    embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
    vectorstore = Chroma.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
except Exception as e:
    print(f"Error during embeddings or vector store creation: {e}")
    raise

# Load LLM
try:
    llm = LlamaCpp(
        model_path=MODEL_PATH,
        temperature=0.2,
        max_tokens=2048,
        top_p=1
    )
except Exception as e:
    print(f"Error loading LLM: {e}")
    raise

# Define the prompt template
template = """
<|context|>
You are a Medical Assistant that follows the instructions and generates accurate responses based on the query and the context provided.
Please be truthful and give direct answers.
</s>
<|user|>
{query}
</s>
<|assistant|>
"""
prompt = ChatPromptTemplate.from_template(template)

# Create the RAG chain
try:
    rag_chain = (
        {"context": retriever, "query": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
except Exception as e:
    print(f"Error creating RAG chain: {e}")
    raise

# API Endpoint
@app.route("/")
def home():
    return "Flask app is running on Render!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        question = data.get("question")
        if not question:
            return jsonify({'error': 'No question provided'}), 400

        # Get response from RAG chain
        response = rag_chain.invoke(question)
        return jsonify({'answer': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port )


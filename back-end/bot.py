from flask import Flask, request, jsonify
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import LlamaCpp
import os

# Initialize Flask app
app = Flask(__name__)

# Configure model and paths
MODEL_PATH = "./BioMistral-7B.Q4_K_M.gguf"  # Ensure this points to your local LlamaCpp model file
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "your_hf_api_key_here")

# Load PDFs and split into chunks
try:
    loader_1 = PyPDFDirectoryLoader("./pdfs/1.pdf")
    loader_2 = PyPDFDirectoryLoader("./pdfs/2.pdf")
    loader_3 = PyPDFDirectoryLoader("./pdfs/3.pdf")
    docs = loader_1.load() + loader_2.load() + loader_3.load()

    if not docs:
        raise ValueError("No documents loaded. Ensure the paths and PDFs are correct.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)

    if not chunks:
        raise ValueError("Document chunks are empty. Check the text splitting logic.")

except Exception as e:
    print(f"Error during document loading and processing: {e}")
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

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

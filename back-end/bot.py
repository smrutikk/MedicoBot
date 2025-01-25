from flask import Flask, request, jsonify
from flask_cors import CORS  # To allow requests from the frontend
from llama_cpp import Llama
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Load model and embeddings
model_path = "/content/drive/MyDrive/Heart-Medico-GUVI/BioMistral-7B.Q4_K_M.gguf"
llm = Llama(model_path=model_path)
embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

# Load PDFs and create a vector store
pdf_directory = "/back-end/pdfs"
loader = PyPDFDirectoryLoader(pdf_directory)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Define the prompt template
template = """
<|context|>
You are a Medical Assistant that follows the instruction and generates an accurate response based on the query and the context provided.
Please be truthful and give direct answers.
</s>
<|user|>
{query}
</s>
<|assistant|>
"""
prompt = ChatPromptTemplate.from_template(template)

# RAG chain
rag_chain = (
    {"context": retriever, "query": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data
        data = request.json
        question = data.get("question")

        # Ensure question is provided
        if not question:
            return jsonify({'error': 'No question provided'}), 400

        # Query the RAG chain
        response = rag_chain.invoke(question)

        return jsonify({'answer': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

from flask import Flask, request, jsonify
from flask_cors import CORS  # To allow requests from the frontend
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
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

# Hugging Face API details
model_name = "TheBloke/Mistral-7B-v0.1-GGUF"  # Update with your model path
hf_token = os.getenv("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")
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)

# Load embeddings
embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

# Load PDFs and create a vector store
pdf_directory = "./pdfs"
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

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data
        data = request.json
        question = data.get("question")

        # Ensure question is provided
        if not question:
            return jsonify({'error': 'No question provided'}), 400

        # Retrieve context from vectorstore
        context_docs = retriever.get_relevant_documents(question)
        context = " ".join([doc.page_content for doc in context_docs])

        # Prepare the prompt for the LLM
        prompt = f"<|context|>\n{context}\n<|user|>\n{question}\n<|assistant|>\n"

        # Generate response using the Hugging Face model
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=2048, temperature=0.7)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return jsonify({'answer': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

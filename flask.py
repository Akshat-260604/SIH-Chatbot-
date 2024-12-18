from flask import Flask, request, jsonify
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import CTransformers

app = Flask(__name__)

def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

extracted_data = load_pdf(r"C:\\Users\\aksha\\OneDrive\\Desktop\\sih_cb\\Q and A.pdf")

def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = text_split(extracted_data)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
docsearch = FAISS.from_texts([t.page_content for t in text_chunks], embeddings)

prompt_template = """
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know the answer at this moment as you are trained on a specific type of data, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

llm = CTransformers(
    model="TheBloke/llama-2-7b-chat-GGML",
    model_type="llama",
    config={'max_new_tokens': 512, 'temperature': 0.8}
)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_question = request.json.get("question", "")
        if not user_question:
            return jsonify({"error": "Question cannot be empty"}), 400
        
        result = qa({"query": user_question})
        response = result["result"]
        return jsonify({"answer": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return "Chatbot Backend is Running!"

if __name__ == "__main__":
    app.run(debug=True)

import streamlit as st
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
from sentence_transformers import SentenceTransformer
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader
import cassio
import time

# Database keys
ASTRA_DB_APPLICATION_TOKEN = "AstraCS:TtGehvnIwWDeKJZzMgDbHWLl:22598a932fde7e17d96f7779732852625f92f6693623046647729e1871c3b4eb"
ASTRA_DB_ID = "a0cc0f28-7626-4d34-b71e-f9386579922b"

# Initialize database connection
cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

# Initialize Hugging Face models
qa_model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
qa_pipeline = pipeline("question-answering", model=qa_model, tokenizer=qa_tokenizer)

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Custom embedding class to handle SentenceTransformer
class CustomEmbedding:
    def __init__(self, model):
        self.model = model
    
    def embed_query(self, text):
        return self.model.encode(text)
    
    def embed_documents(self, texts):
        return self.model.encode(texts)

custom_embedding = CustomEmbedding(embedding_model)

astra_vector_store = Cassandra(
    embedding=custom_embedding,
    table_name="qa_mini_demo",
    session=None,
    keyspace=None,
)

# Streamlit app configuration
st.set_page_config(page_title="PDF Question Answering System", page_icon="📄", layout="wide")

st.title("📄 PDF Question Answering System")

st.markdown(
    """
    <style>
    .main { 
        background-color: #black;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    </style>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Display a progress bar
    progress_bar = st.progress(0)
    st.info("Reading and processing PDF...")

    # Read text from PDF
    pdfreader = PdfReader(uploaded_file)
    raw_text = ''
    num_pages = len(pdfreader.pages)
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            raw_text += content
        progress_bar.progress((i + 1) / num_pages)
        time.sleep(0.1)  # Simulate some delay for better visualization

    st.success("PDF content loaded successfully!")

    # Split the text
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)
    
    # Add texts to the vector store
    astra_vector_store.add_texts(texts[:50])
    st.write(f"Inserted {len(texts[:50])} text chunks into the vector store.")

    astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)
    
    st.write("You can now ask questions based on the PDF content.")
    
    # Input field for user question
    query_text = st.text_input("Enter your question:")

    if query_text:
        st.subheader(f"**QUESTION:** {query_text}")
        answer = qa_pipeline(question=query_text, context=raw_text)["answer"]
        st.subheader(f"**ANSWER:** {answer}")
        # Clear the input field after showing the answer
        st.text_input("Enter your question:", value="", key="new_question")

else:
    st.info("Please upload a PDF file to get started.")

# Additional Styling and Footer
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        color: black;
        text-align: center;
        padding: 10px;
        font-size: 12px;
    }
    </style>
    <div class="footer">
        <p>Developed by [Abhishek]. Powered by Streamlit and Hugging Face Transformers.</p>
    </div>
    """,
    unsafe_allow_html=True
)

import streamlit as st
import os
import io
import PyPDF2
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document  # Import Document class

# Directly set your API keys here
GROQ_API_KEY = "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"
GOOGLE_API_KEY = "AIzaSyCbfZh072K8_4csXUOeiDsqLT9u-kmB6Y8"

# Initialize session state variables
if "vectors" not in st.session_state:
    st.session_state.vectors = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "docs" not in st.session_state:
    st.session_state.docs = None
if "final_documents" not in st.session_state:
    st.session_state.final_documents = None

# Initialize the language model
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="Llama3-8b-8192")

# Define prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Question: {input}
    """
)

def vector_embedding(uploaded_files):
    try:
        # Initialize embeddings if not done
        if st.session_state.vectors is None:
            st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

            texts = []  # Store text content here
            for uploaded_file in uploaded_files:
                with io.BytesIO(uploaded_file.read()) as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    text = ""
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text
                    texts.append(text)  # Append the extracted text

            # Prepare documents in the required format
            st.session_state.docs = [Document(page_content=text, metadata={"filename": uploaded_file.name}) for text in texts]
            
            # Create text splitter and split documents
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.final_documents = text_splitter.split_documents(st.session_state.docs)
            
            # Initialize the vector store
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
    except Exception as e:
        st.error(f"Error processing documents: {e}")

# Streamlit file uploader for PDF documents
uploaded_files = st.file_uploader("Upload PDF documents", accept_multiple_files=True, type="pdf")

if st.button("Documents Embedding") and uploaded_files:
    with st.spinner("Processing documents..."):
        vector_embedding(uploaded_files)
    st.success("Vector Store DB Is Ready")

prompt1 = st.text_input("Enter Your Question From Documents")

if prompt1 and st.session_state.vectors is not None:
    with st.spinner("Generating answer..."):
        try:
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            response = retrieval_chain.invoke({'input': prompt1})
            st.write("Answer:", response['answer'])

            # Document similarity search section
            with st.expander("Document Similarity Search"):
                for doc in response.get("context", []):
                    st.write(doc.page_content)
                    st.write("--------------------------------")
        except Exception as e:
            st.error(f"Error generating answer: {e}")

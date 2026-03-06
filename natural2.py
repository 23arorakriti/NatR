import streamlit as st
import os
import base64
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ---------------------------
# Load API Key
# ---------------------------
load_dotenv(".env")
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("❌ GOOGLE_API_KEY not found in .env file")
    st.stop()

# ---------------------------
# Background Image
# ---------------------------
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

image_base64 = get_base64_image("back2.jpeg")

# ---------------------------
# Gemini Flash LLM
# ---------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    google_api_key=api_key,
    temperature=0.7
)

# ---------------------------
# Sentence Transformer Embeddings
# ---------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectordb_file_path = "faiss_index"

# ---------------------------
# Create FAISS Vector DB
# ---------------------------
def create_vector_db():

    loader = CSVLoader(file_path="NR.csv")

    documents = loader.load()

    vectordb = FAISS.from_documents(
        documents,
        embeddings
    )

    vectordb.save_local(vectordb_file_path)

# ---------------------------
# Build RAG Chain
# ---------------------------
def get_qa_chain():

    if not os.path.exists(vectordb_file_path):

        st.info("Vector database not found. Creating now...")
        create_vector_db()
        st.success("Vector database created successfully!")

    vectordb = FAISS.load_local(
        vectordb_file_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectordb.as_retriever(
        search_kwargs={"k": 5}
    )

    prompt_template = """

You are an expert in natural remedies and alternative medicine.

Use the CONTEXT below to answer the user’s question clearly.

CONTEXT:
{context}

QUESTION:
{question}

Provide your answer using:

1. Overview
2. Natural Remedies
3. Practical Recommendations
4. Additional Suggestions

Rules:
- Do not mention the document
- If disease not found, suggest closest natural remedies
- If irrelevant question → say "I don't know"

"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

# ---------------------------
# Streamlit Page Setup
# ---------------------------
st.set_page_config(page_title="NatureMend", layout="centered")

st.markdown(
    f"""
<style>

.stApp {{
background-image: url("data:image/jpeg;base64,{image_base64}");
background-size: cover;
}}

.stButton>button {{
background-color: #1fdd93;
color: white;
border-radius: 10px;
}}

.stTextInput>div>div>input {{
border: 2px solid #1fdd93;
border-radius: 8px;
}}

</style>
""",
    unsafe_allow_html=True
)

# ---------------------------
# UI
# ---------------------------
st.title("🌿 NatureMend")
st.caption("AI-powered natural remedies assistant (Gemini Flash)")

chain = get_qa_chain()

question = st.text_input(
    "Ask your question about natural remedies or health conditions:"
)

if st.button("Submit Question"):

    if not question:
        st.warning("Please enter a question.")

    else:

        with st.spinner("Generating answer..."):

            try:

                response = chain.invoke(question)

                st.markdown("### 🌼 Answer")
                st.write(response)

            except Exception as e:

                st.error(e)
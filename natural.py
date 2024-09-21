import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from hello import api_key  
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_core.runnables import RunnablePassthrough
import base64

os.environ["GOOGLE_API_KEY"] = api_key

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
image_path = "back2.jpeg"  

image_base64 = get_base64_image(image_path)
llm = GoogleGenerativeAI(google_api_key=api_key, model='gemini-1.5-flash', temperature=0.9)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
instructor_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", task_type="retrieval_query"
)

vectordb_file_path = "faiss_index"
def create_vector_db():
    loader = CSVLoader(file_path="NR.csv", source_column="Condition")
    data = loader.load()
    vectordb = FAISS.from_documents(documents=data, embedding=instructor_embeddings)
    vectordb.save_local(vectordb_file_path)

def get_qa_chain():
    
    if not os.path.exists(vectordb_file_path):
        st.write("Vector database file not found. Creating now...")
        create_vector_db()
        st.write("Vector database created successfully.") 
  
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings, allow_dangerous_deserialization=True)

    retriever = vectordb.as_retriever(score_threshold=0.7, top_k=5)


    prompt_template = """
  You are an expert in natural remedies for various diseases. Based on the provided context, generate a detailed and structured response to the user's question.

 Ensure the response is clear, practical, and includes actionable recommendations. Avoid phrases like "the document says" and focus on providing a natural and conversational answer. If the question involves a disease, include relevant natural remedies.

 CONTEXT: {context}

 QUESTION: {question}

 Answer:

 1. **Overview**: Briefly summarize the topic or disease if applicable.
 2. **Natural Remedies**: List and explain natural remedies related to the disease or condition mentioned in the question.
 3. **Practical Recommendations**: Provide practical advice on how to use these remedies, including preparation and dosage.
 4. **Additional Suggestions**: Offer any additional tips or lifestyle changes that could be beneficial.
 5. **If DISEASE Not Found**: Find the most relevant natural remedies and give output. Do not mention that it is not found in documentation.
 6. **If Irrelevant: Answer with "I Don't know".
"""

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    return chain

st.set_page_config(page_title="NatureMend", layout="centered")
st.markdown(
    f"""
    <style>
    .main {{
        background-color: #F5F5F5;
        padding: 20px;
        background-image: url("data:image/jpg;base64,{image_base64}");
        background-size: cover;
        font-family: 'Great Vibes', cursive;
    }}
    .stButton>button {{
        background-color: #1fdd93;
        color: white;
        border: none;
        padding: 10px 24px;
        text-align: center;
        font-size: 16px;
        margin: 4px 2px;
        transition-duration: 0.4s;
        cursor: pointer;
        border-radius: 12px;
    }}
    .stButton>button:hover {{
        background-color: white;
        color: black;
        border: 2px solid #1fdd93;
    }}
    .stTextInput>div>div>input {{
        border: 2px solid #1fdd93;
        border-radius: 8px;
    }}

    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌿 NatureMend")


chain = get_qa_chain()

question = st.text_input("Ask your question about natural remedies or conditions:")

if st.button("Submit Question"):
    if question:
        with st.spinner("Generating your answer..."):
            
                response = chain.invoke(question)
                if isinstance(response, str):
                   
                    st.markdown("### 🌼 **Answer**")
                    st.write(response)
                else:
                    
                    st.markdown("### 🌼 **Answer**")
                    st.write(f"**Overview**: {response.get('Overview', 'Not available')}")
                    st.write(f"**Natural Remedies**: {response.get('Natural Remedies', 'Not available')}")
                    st.write(f"**Practical Recommendations**: {response.get('Practical Recommendations', 'Not available')}")
                    st.write(f"**Additional Suggestions**: {response.get('Additional Suggestions', 'Not available')}")

            
    else:
        st.write("Please enter a question.")

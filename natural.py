import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from hello import api_key  # Assuming this is where the API key is stored
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_core.runnables import RunnablePassthrough

# Set the API key in the environment
os.environ["GOOGLE_API_KEY"] = api_key

# Initialize the Google Generative AI model for question-answering
llm = GoogleGenerativeAI(google_api_key=api_key, model='gemini-1.5-flash', temperature=0.9)

# Initialize embeddings model for retrieval
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Create another embedding for query retrieval
instructor_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", task_type="retrieval_query"
)

# Path to save/load the FAISS vector database
vectordb_file_path = "faiss_index"

# Function to create the vector database from a CSV file
def create_vector_db():
    # Load the CSV file, assuming there's a column called 'Condition' containing the relevant data
    loader = CSVLoader(file_path="NR.csv", source_column="Condition")
    data = loader.load()

    # Create the vector store using FAISS and store locally
    vectordb = FAISS.from_documents(documents=data, embedding=instructor_embeddings)
    vectordb.save_local(vectordb_file_path)

# Function to set up the QA chain
def get_qa_chain():
    # Check if the vector database exists; if not, create it
    if not os.path.exists(vectordb_file_path):
        st.write("Vector database file not found. Creating now...")
        create_vector_db()
        st.write("Vector database created successfully.")
    
    # Load the vector database
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings, allow_dangerous_deserialization=True)

    # Set up the retriever with a score threshold to filter relevant documents and return top 5 chunks
    retriever = vectordb.as_retriever(score_threshold=0.7, top_k=5)

    # Define the prompt template
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
5. **If DISEASE Not Found**: Find the most relevant natural remedies and give output
6. **If Irrelevant:Answer with "I Don't know"
"""

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Create a RAG chain using the retriever, prompt, and LLM
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    return chain

# Main entry point for running the app
if __name__ == "__main__":
    # Streamlit interface
    st.title("NATURAL")

    # Set up the QA chain
    chain = get_qa_chain()
    
    # Input for user's question
    question = st.text_input("Ask your question here:")

    # If the user submits a question, run the QA chain
    if st.button("Submit Question"):
        if question:
            try:
                response = chain.invoke(question)
                st.write(f"Answer: {response}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.write("Please enter a question.")

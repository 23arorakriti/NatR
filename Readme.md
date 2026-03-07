# 🌿 NatureMend

NatureMend is an AI-powered assistant that provides **natural remedies and health suggestions** for various conditions using **Retrieval-Augmented Generation (RAG)**.

The application combines **Gemini Flash (LLM)** with **Sentence Transformer embeddings** and **FAISS vector search** to retrieve relevant information from a curated dataset of natural remedies.

Users can ask questions about diseases, symptoms, or health concerns and receive structured answers including remedies and practical recommendations.

---

# 🚀 Live Demo

🔗 Streamlit App:  https://naturalremedies.streamlit.app/
---

# 📌 Project Overview

NatureMend helps users explore **natural remedies and alternative medicine suggestions** through an AI-powered interface.

The system:

1. Stores remedy data in a CSV dataset.
2. Converts the data into vector embeddings.
3. Stores embeddings in a FAISS vector database.
4. Retrieves relevant context based on user questions.
5. Uses Gemini Flash to generate structured answers.

This architecture improves response quality by grounding the model in **domain-specific knowledge**.

---

# 🧠 Architecture

The project follows a **Retrieval-Augmented Generation (RAG)** workflow.

User Question  
↓  
Embedding Generation (Sentence Transformers)  
↓  
Vector Search (FAISS)  
↓  
Retrieve Relevant Context  
↓  
Prompt Template  
↓  
Gemini Flash LLM  
↓  
Generated Answer

---

# ✨ Features

- AI-powered natural remedy suggestions
- Retrieval-Augmented Generation (RAG)
- Context-aware responses
- Fast semantic search with FAISS
- Streamlit interactive interface
- CSV-based knowledge dataset

---

# 🛠 Tech Stack

### Programming Language
- Python

### LLM
- Gemini Flash (`gemini-flash-latest`)

### Embeddings
- Sentence Transformers (`all-MiniLM-L6-v2`)

### Frameworks
- LangChain
- Streamlit

### Vector Database
- FAISS

### Data Handling
- Pandas
- NumPy

# 💬 Example Queries

Users can ask questions like:

- What are natural remedies for cough?
- How can I reduce inflammation naturally?
- Home remedies for digestive issues
- Natural ways to boost immunity

The AI will provide structured responses including remedies and practical recommendations.

# 👩‍💻 Author

**Kriti Arora**


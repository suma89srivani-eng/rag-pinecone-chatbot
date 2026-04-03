import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np

# -------------------------------
# Page Settings
# -------------------------------
st.set_page_config(page_title="RAG Chatbot using Pinecone", page_icon="📘", layout="centered")

# -------------------------------
# Load embedding model
# -------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# -------------------------------
# Convert text to embeddings
# -------------------------------
def generate_embeddings(texts, model):
    return model.encode(texts)


# -------------------------------
# Load and prepare documents
# -------------------------------
@st.cache_data
def load_documents(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]
    return lines


# -------------------------------
# Simulated Pinecone-style retrieval
# -------------------------------
def retrieve(query, model, documents, doc_embeddings, top_k=2):
    query_embedding = generate_embeddings([query], model)[0]

    similarities = []
    for i, emb in enumerate(doc_embeddings):
        score = np.dot(query_embedding, emb) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(emb)
        )
        similarities.append((score, documents[i]))

    similarities = sorted(similarities, reverse=True, key=lambda x: x[0])
    return [doc for _, doc in similarities[:top_k]]


# -------------------------------
# Streamlit UI
# -------------------------------
def main():
    st.title("📘 RAG Chatbot using Pinecone")
    st.write("Upload a text file and ask questions based on semantic retrieval.")

    uploaded_file = st.file_uploader("Upload your sample_data.txt file", type="txt")

    if uploaded_file is not None:
        filepath = "temp.txt"

        with open(filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success("File uploaded successfully ✅")

        # Load model and documents
        model = load_model()
        documents = load_documents(filepath)
        doc_embeddings = generate_embeddings(documents, model)

        st.info(f"Loaded {len(documents)} document entries")

        # User query
        query = st.text_input("Ask a question:")

        if query:
            results = retrieve(query, model, documents, doc_embeddings)

            st.subheader("📌 Answer:")
            for i, res in enumerate(results, 1):
                st.write(f"**Result {i}:** {res}")


# -------------------------------
# Run app
# -------------------------------
if __name__ == "__main__":
    main()

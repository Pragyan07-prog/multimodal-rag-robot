from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def run_retrieval_pipeline():
    db_dir = "faiss_index"
    
    if not os.path.exists(db_dir):
        print(f"❌ Error: Local vector index '{db_dir}' not found. Please run ingest.py first!")
        return

    # Load local embeddings and FAISS index
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # allow_dangerous_deserialization is required to open local pickled FAISS files safely
    vector_db = FAISS.load_local(db_dir, embeddings, allow_dangerous_deserialization=True)
    
    # ---- Step 8: Accept User Question ----
    print("\n🤖 Local FAISS RAG Pipeline Active.")
    user_question = input("🙋 Ask a question about your document: ")
    
    # ---- Step 9 & 10: Convert Question into Embedding & Perform Similarity Search ----
    print("⏳ Transforming question to vector and evaluating similarity matches...")
    retrieved_results = vector_db.similarity_search(user_question, k=2) # Fetches top 2 context matches
    
    # ---- Step 11: Display Retrieved Information ----
    print("\n================== [Step 11] RETRIEVED INFORMATION ==================")
    for i, doc in enumerate(retrieved_results):
        page_num = doc.metadata.get("page", 0) + 1
        print(f"\n📄 Match #{i+1} (Source: Document Page {page_num}):")
        print(doc.page_content)
        print("-" * 65)
    print("=====================================================================")

if __name__ == "__main__":
    run_retrieval_pipeline()
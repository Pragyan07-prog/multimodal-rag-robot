from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def run_ingestion():
    pdf_path = os.path.join("data", "bts_history.pdf")
    db_dir = "faiss_index"
    
    # ---- Step 4: Load PDF Document ----
    print("⏳ [Step 4] Loading PDF Document...")
    if not os.path.exists(pdf_path):
        print(f"❌ Error: Cannot find '{pdf_path}' in the data folder. Please verify the file name!")
        return
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"✅ Success: Extracted text from all {len(documents)} pages.\n")
    
    # ---- Step 5: Split Text into Chunks ----
    print("⏳ [Step 5] Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, 
        chunk_overlap=150  # Overlap preserves context between chunks
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Success: Verified total number of chunks created: {len(chunks)}\n")
    
    # ---- Step 6: Generate Embeddings ----
    print("⏳ [Step 6] Loading pre-trained embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("✅ Success: Embeddings model initialized and verified successfully.\n")
    
    # ---- Step 7: Create Vector Database (FAISS) ----
    print("⏳ [Step 7] Storing chunk embeddings inside FAISS Vector Database...")
    vector_db = FAISS.from_documents(chunks, embeddings)
    
    # Save the index locally as required by your workflow
    vector_db.save_local(db_dir)
    print(f"✅ Success: Vector database created and saved locally to '{db_dir}/'\n")
    print("🎉 Steps 1-7 (Phase 1 Ingestion Workflow) completed perfectly!")

if __name__ == "__main__":
    run_ingestion()
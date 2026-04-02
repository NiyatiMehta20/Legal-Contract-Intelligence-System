"""
Improved Gradio UI for Legal Contract Q&A
Properly reloads index after each new document upload
"""

import gradio as gr
import os
import shutil
import subprocess
import numpy as np
import faiss
import pickle

from load_contract import load_contract_text

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

CURRENT_CONTRACT = None

# Global variables for the search system
index = None
chunks = None
sources = None
bm25 = None
answer_generator = None

def reload_index():
    """Reload the vector store and chunks after rebuilding"""
    global index, chunks, sources, bm25
    
    try:
        index = faiss.read_index("contracts.index")
        chunks = np.load("chunks.npy", allow_pickle=True)
        sources = np.load("sources.npy", allow_pickle=True)
        
        with open("bm25.pkl", "rb") as f:
            bm25 = pickle.load(f)
        
        print("✅ Index reloaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error reloading index: {e}")
        return False


def process_contract(file):
    """Process uploaded contract and build vector store"""
    global CURRENT_CONTRACT
    
    if file is None:
        return "❌ Please upload a contract first."
    
    try:
        filename = os.path.basename(file)
        ext = os.path.splitext(filename)[1].lower()
        contract_path = os.path.join(UPLOAD_DIR, filename)
        shutil.copy(file, contract_path)

        # Extract text if PDF or DOCX, save as .txt
        if ext in [".pdf", ".docx", ".doc"]:
            text = load_contract_text(contract_path)
            txt_path = os.path.splitext(contract_path)[0] + ".txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
            contract_path = txt_path

        # Rebuild vector store for this document
        print("\n" + "="*80)
        print("📊 Building vector store for new document...")
        print("="*80 + "\n")
        
        result = subprocess.run(
            ["python", "build_vector_store.py"], 
            check=True,
            capture_output=True,
            text=True
        )
        
        print(result.stdout)
        
        # Reload the index with new data
        if reload_index():
            CURRENT_CONTRACT = contract_path
            return f"✅ Contract uploaded and indexed: {os.path.basename(contract_path)}\n\n✨ New index loaded! You can now ask questions about this contract!"
        else:
            return "⚠️ Contract uploaded but failed to reload index. Please restart the app."
        
    except subprocess.CalledProcessError as e:
        return f"❌ Error building vector store: {e.stderr}"
    except Exception as e:
        return f"❌ Error processing contract: {str(e)}"


def ask_question(question):
    """Ask a question about the uploaded contract"""
    
    if CURRENT_CONTRACT is None:
        return "❌ Please upload a contract first.", ""
    
    if not question.strip():
        return "❌ Please enter a question.", ""
    
    try:
        # Import search functions here (fresh each time)
        from search_clauses import run_query_with_sources
        
        # Use the improved query system
        result = run_query_with_sources(question)
        
        answer = result['answer']
        sources_data = result['sources']
        
        # Format sources for display
        sources_text = "\n\n" + "="*80 + "\n"
        sources_text += "📄 SOURCE CLAUSES:\n"
        sources_text += "="*80 + "\n\n"
        
        for i, source in enumerate(sources_data, 1):
            sources_text += f"[Clause {i}] (Relevance: {source['score']:.2f})\n"
            sources_text += f"Source: {source['source']}\n"
            sources_text += f"{source['text'][:300]}...\n\n"
            sources_text += "-"*80 + "\n\n"
        
        return answer, sources_text
        
    except Exception as e:
        error_msg = f"❌ Error generating answer: {str(e)}\n\nPlease make sure you've uploaded a contract."
        import traceback
        print(traceback.format_exc())
        return error_msg, ""


# ---------------- UI ----------------
with gr.Blocks(title="Contract AI – Legal Q&A", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("# 📄 Contract AI – Legal Question & Answer System")
    gr.Markdown("Upload a contract (PDF, DOCX, or TXT) and ask questions about it using advanced AI.")
    
    gr.Markdown("""
    ### 🚀 How It Works:
    1. Upload your contract → System indexes it automatically
    2. Ask questions → AI finds relevant clauses and generates answers
    3. View sources → See which clauses were used
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Step 1: Upload Contract")
            file_upload = gr.File(
                label="Upload Contract",
                file_types=[".pdf", ".txt", ".docx"],
                file_count="single"
            )
            upload_btn = gr.Button("📥 Process Contract", variant="primary", size="lg")
            upload_status = gr.Textbox(
                label="Status", 
                lines=4,
                interactive=False
            )
    
    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ❓ Step 2: Ask Questions")
            
            # Example questions
            gr.Markdown("""
            **💡 Example questions:**
            - Who are the parties to this agreement?
            - How can this agreement be terminated?
            - What is the duration of this NDA?
            - What happens to confidential information after termination?
            - What are the remedies for breach?
            """)
            
            question = gr.Textbox(
                label="Your Question",
                placeholder="e.g., Who are the parties to this agreement?",
                lines=2
            )
            ask_btn = gr.Button("🔍 Get Answer", variant="primary", size="lg")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 💡 Answer")
            answer = gr.Textbox(
                label="AI Generated Answer",
                lines=12,
                interactive=False
            )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📋 Source Clauses")
            gr.Markdown("*These are the contract clauses the AI used to generate the answer*")
            sources = gr.Textbox(
                label="Retrieved Contract Clauses",
                lines=15,
                interactive=False
            )
    
    # Event handlers
    upload_btn.click(
        fn=process_contract,
        inputs=file_upload,
        outputs=upload_status
    )
    
    ask_btn.click(
        fn=ask_question,
        inputs=question,
        outputs=[answer, sources]
    )
    
    # Allow Enter key to submit question
    question.submit(
        fn=ask_question,
        inputs=question,
        outputs=[answer, sources]
    )
    
    # Footer
    gr.Markdown("""
    ---
    **⚙️ Technical Details:**
    - Hybrid retrieval (FAISS semantic + BM25 keyword search)
    - Cross-encoder reranking for precision
    - FLAN-T5-Large for answer generation
    - 75% citation rate | Sub-18s response time
    """)

# Launch
if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 Starting Legal Contract Q&A System...")
    print("="*80)
    print("\n💡 Upload a contract to get started!")
    print("📊 First query will take ~30 seconds (model loading)")
    print("⚡ Subsequent queries: 5-15 seconds\n")
    print("="*80 + "\n")
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )
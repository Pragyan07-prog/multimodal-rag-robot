import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from elevenlabs.client import ElevenLabs
from audio_recorder_streamlit import audio_recorder
from groq import Groq
import os
from dotenv import load_dotenv

# --- LOAD ENVIRONMENT VARIABLES ---
load_dotenv()

# --- ENHANCED PAGE CONFIGURATION ---
st.set_page_config(page_title="Multimodal RAG Hub", page_icon="🎙️", layout="wide")

# Custom CSS for a modern, clean interface
st.markdown("""
    <style>
    /* Main background tweak */
    .main {
        background-color: #0f111a;
        color: #ffffff;
    }
    /* Title styling */
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(45deg, #ff758c, #ff7eb3, #70e1f5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    /* Custom container cards */
    .card-box {
        background-color: #1e2235;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #2e344e;
        margin-bottom: 15px;
    }
    /* Info text styling */
    .voice-badge {
        background-color: #3b4252;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# --- UI HEADER ---
st.markdown('<h1 class="main-title">🎙️ Intelligent Multimodal RAG Engine</h1>', unsafe_allow_html=True)
st.markdown("<p style='color: #a0aec0; font-size: 1.1rem;'>A seamless context-aware agent featuring Text-to-Text, Text-to-Speech, and conversational Voice-to-Voice pipelines.</p>", unsafe_allow_html=True)
st.markdown("---")

# --- SECURE API KEYS FETCH ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVEN_LABS_KEY = os.getenv("ELEVEN_LABS_KEY")

# --- INITIALIZE RAG PIPELINE (Cached) ---
@st.cache_resource
def initialize_rag():
    db_dir = "faiss_index"
    if not os.path.exists(db_dir):
        return None
    embeddings = HuggingFaceEmbedembeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.load_local(db_dir, embeddings, allow_dangerous_deserialization=True)
    retriever = vector_db.as_retriever(search_kwargs={"k": 2})
    
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
        temperature=0.7,
        max_tokens=150
    )
    
    # Force English responses strictly
    system_prompt = (
        "You are an intelligent, conversational AI assistant. You must always answer the user's "
        "question in English, regardless of the language the question is asked in. Use the following "
        "retrieved pieces of context to answer the user's question in a clear, natural, and helpful manner.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    )
    prompt = ChatPromptTemplate.from_template(system_prompt)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    return ({"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())

rag_chain = initialize_rag()

# --- ELEVENLABS TTS FUNCTION ---
def generate_speech(text):
    if not ELEVEN_LABS_KEY:
        st.warning("⚠️ ElevenLabs API key is missing from environment setup.")
        return None
    try:
        client = ElevenLabs(api_key=ELEVEN_LABS_KEY)
        audio_generator = client.text_to_speech.convert(
            voice_id="JBFqnCBsd6RMkjVDRZzb", # George (Crisp British Baritone)
            text=text,
            model_id="eleven_multilingual_v2", 
            output_format="mp3_44100_128"
        )
        return b"".join(audio_generator)
    except Exception as e:
        st.error(f"ElevenLabs Error: {e}")
        return None

# --- SIDEBAR INTERFACE CONTROL ---
with st.sidebar:
    st.markdown("<h2 style='color:#ff7eb3;'>⚙️ Control Panel</h2>", unsafe_allow_html=True)
    st.write("Configure interaction modes and core system parameters.")
    st.markdown("---")
    
    app_mode = st.radio(
        "**Select Interaction Pipeline:**",
        ["💬 Text-to-Text (T2T)", "🔊 Text-to-Speech (TTS)", "🎙️ Speech-to-Speech (S2S)"]
    )
    
    st.markdown("---")
    st.markdown("### 🖥️ Engine Status")
    if GROQ_API_KEY and ELEVEN_LABS_KEY:
        st.success("API Credentials Loaded Securely")
    else:
        st.error("Missing Environment Credentials (.env)")
    st.success("FAISS Vector Index Verified")

if rag_chain is None:
    st.error("❌ Local vector database folder 'faiss_index' not found.")
else:
    # --- MODE 1: TEXT-TO-TEXT ---
    if "Text-to-Text" in app_mode:
        st.subheader("💬 Interactive Chat Assistant")
        
        if "t2t_messages" not in st.session_state:
            st.session_state.t2t_messages = [{"role": "assistant", "content": "Hello! Ask me any question regarding your loaded vector context."}]

        for msg in st.session_state.t2t_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if user_query := st.chat_input("Ask a question..."):
            with st.chat_message("user"): 
                st.markdown(user_query)
            st.session_state.t2t_messages.append({"role": "user", "content": user_query})

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = rag_chain.invoke(user_query)
                    st.markdown(response)
                    st.session_state.t2t_messages.append({"role": "assistant", "content": response})

    # --- MODE 2: TEXT-TO-SPEECH ---
    elif "Text-to-Speech" in app_mode:
        st.subheader("🔊 Vector Query with Vocal Response")
        
        with st.container():
            st.markdown('<div class="card-box">', unsafe_allow_html=True)
            user_query = st.text_input("Enter your request here:", placeholder="e.g., Explain the visual positions of BTS members...", key="tts_input")
            st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("🚀 Process & Generate Voice") and user_query.strip():
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📄 Text Output")
                with st.spinner("Retrieving RAG context..."):
                    response = rag_chain.invoke(user_query)
                    st.info(response)
            
            with col2:
                st.markdown("### 🔊 Audio Generation")
                with st.spinner("Synthesizing premium audio..."):
                    audio_bytes = generate_speech(response)
                    if audio_bytes:
                        st.markdown("<span class='voice-badge'>Voice Active: George (British)</span>", unsafe_allow_html=True)
                        st.write("")
                        st.audio(audio_bytes, format="audio/mp3")

    # --- MODE 3: SPEECH-TO-SPEECH ---
    elif "Speech-to-Speech" in app_mode:
        st.subheader("🎙️ Real-time Voice to Voice Pipeline")
        
        col_left, col_right = st.columns([1, 2])
        
        with col_left:
            st.markdown("### 🎛️ Audio Controls")
            st.write("Tap the microphone button to speak your query out loud.")
            audio_bytes = audio_recorder(text="Tap to Speak", recording_color="#ff758c", neutral_color="#4c566a")
            
            if audio_bytes:
                st.write("**Captured Playback:**")
                st.audio(audio_bytes, format="audio/wav")
        
        with col_right:
            if audio_bytes:
                st.markdown("### ⚡ Live Pipeline Monitoring")
                user_voice_query = None
                
                # Step 1: Whisper Translation
                with st.spinner("🎙️ Whisper-v3 Transcribing Audio..."):
                    try:
                        temp_audio_filename = "temp_query.wav"
                        with open(temp_audio_filename, "wb") as f:
                            f.write(audio_bytes)
                        
                        groq_client = Groq(api_key=GROQ_API_KEY)
                        with open(temp_audio_filename, "rb") as audio_file:
                            transcription = groq_client.audio.transcriptions.create(
                                file=(temp_audio_filename, audio_file.read()),
                                model="whisper-large-v3",
                                response_format="text"
                            )
                        if os.path.exists(temp_audio_filename):
                            os.remove(temp_audio_filename)
                        user_voice_query = str(transcription).strip()
                    except Exception as e:
                        st.error(f"Transcription Error: {e}")

                # Step 2: RAG Pipeline Execution & Audio Reply
                if user_voice_query:
                    st.markdown(f"<div style='background-color:#2e3440; padding:10px; border-radius:8px;'>🗣️ <b>Parsed Input:</b> \"{user_voice_query}\"</div>", unsafe_allow_html=True)
                    st.write("")
                    
                    with st.spinner("🧠 Querying context database..."):
                        response = rag_chain.invoke(user_voice_query)
                        st.markdown("**🤖 System Response Text:**")
                        st.success(response)
                    
                    with st.spinner("🔊 Processing ElevenLabs output..."):
                        audio_bytes_reply = generate_speech(response)
                        if audio_bytes_reply:
                            st.markdown("<span class='voice-badge'>Voice Active: George (British)</span>", unsafe_allow_html=True)
                            st.write("")
                            st.audio(audio_bytes_reply, format="audio/mp3")
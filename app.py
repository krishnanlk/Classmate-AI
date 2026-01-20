import streamlit as st
import os
import json
from datetime import datetime
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import wikipedia
from gemini_chat import gemini_chat
from connect import load_all_lectures

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Classmate AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== STORAGE ==================
BASE_DIR = "cloud_storage"
CHAT_DIR = "chat_history"
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(CHAT_DIR, exist_ok=True)

# ================== CHAT PERSISTENCE ==================
def chat_file(user_id):
    return os.path.join(CHAT_DIR, f"{user_id}.json")

def load_chat(user_id):
    path = chat_file(user_id)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_chat(user_id, history):
    with open(chat_file(user_id), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

# ================== GLOBAL STYLE ==================
st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }

.stApp {
    background: radial-gradient(circle at top, #0b0f19, #000000);
    color: #e5e7eb;
}

.block-container {
    padding-top: 2rem;
    max-width: 100% !important;
}

/* NAVBAR */
.navbar {
    max-width: 900px;
    margin: 30px auto;
    padding: 14px 20px;
    background: #1f2430;
    border-radius: 14px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.nav-btn button {
    background: #ff4d4f !important;
    color: white !important;
    border-radius: 10px !important;
    padding: 8px 18px !important;
    font-weight: 600;
}

.auth-box {
    max-width: 900px;
    margin: auto;
    padding-top: 30px;
}

.tabs {
    display: flex;
    gap: 30px;
    border-bottom: 1px solid #374151;
    margin-bottom: 20px;
}

.tab-active {
    color: #ff4d4f;
    border-bottom: 2px solid #ff4d4f;
    padding-bottom: 8px;
    font-weight: 600;
}

.input-label {
    margin-top: 18px;
    color: #e5e7eb;
    font-size: 14px;
}

/* INPUTS */
input {
    background: #2a2f3a !important;
    border-radius: 10px !important;
    padding: 14px !important;
    border: none !important;
    color: white !important;
}

/* ================== SELECTBOX FIX ================== */

/* Main select container */
div[data-baseweb="select"] > div {
    background-color: #2a2f3a !important;
    border-radius: 10px !important;
    border: none !important;
}

/* Remove left segmented box */
div[data-baseweb="select"] > div > div {
    background: transparent !important;
}

/* Text color */
div[data-baseweb="select"] span {
    color: white !important;
}

/* Dropdown arrow */
div[data-baseweb="select"] svg {
    fill: #9ca3af !important;
}

/* Hover & focus */
div[data-baseweb="select"]:hover > div {
    background-color: #323846 !important;
}

</style>
""", unsafe_allow_html=True)

# ================== SESSION INIT ==================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user = None
    st.session_state.role = None

if "page" not in st.session_state:
    st.session_state.page = "home"

if "page" not in st.query_params:
    st.query_params["page"] = "home"


# ================== IMAGE-EXACT NAVBAR ==================
if not st.session_state.logged_in:

    st.markdown("""
    <style>
    .top-nav {
        max-width: 780px;
        margin: 30px auto;
        background: #1f2430;
        border-radius: 14px;
        padding: 8px;
        position: relative;
        display: flex;
        align-items: center;
    }

    /* MOVING GLOW */
    .nav-glow {
        position: absolute;
        height: 42px;
        width: 120px;
        background: #ff4d4f;
        border-radius: 10px;
        left: 8px;
        top: 8px;
        box-shadow: 0 0 18px rgba(255,77,79,0.8);
        transition: left 0.35s ease;
        z-index: 0;
    }

    /* NAV ITEMS */
    .nav-item {
        position: relative;
        z-index: 1;
        padding: 10px 20px;
        color: white;
        font-size: 14px;
        font-weight: 500;
        text-decoration: none;
        border-radius: 10px;
        cursor: pointer;
    }

    /* PUSH LOGIN TO RIGHT */
    .nav-login {
        margin-left: auto;
    }

    /* HOVER → MOVE GLOW */
    .nav-home:hover ~ .nav-glow {
        left: 8px;
    }

    .nav-login:hover ~ .nav-glow {
        left: calc(100% - 128px);
    }
    </style>

    <div class="top-nav">
        <a href="?page=home" class="nav-item nav-home">🏠 Home Page</a>
        <a href="?page=auth" class="nav-item nav-login"> Login Page</a>
        <div class="nav-glow"></div>
    </div>
    """, unsafe_allow_html=True)

    if "page" in st.query_params:
        st.session_state.page = st.query_params["page"]

# ================== HOME PAGE ==================
if not st.session_state.logged_in and st.session_state.page == "home":

    st.markdown("""
    <h1 style="color:#ff4d4f;">Classmate AI</h1>

    <p style="font-size:16px; color:#cbd5e1;">
    Classmate AI is an intelligent AI-powered classroom assistant designed to capture,
    organize, and enhance classroom learning.
    </p>

    <div style="
        background:#1f2430;
        padding:18px;
        border-radius:12px;
        margin-top:20px;
    ">
    <ul style="font-size:16px; line-height:1.8; color:#e5e7eb;">
        <li>🎥 Automatic lecture recording</li>
        <li>☁️ Secure cloud storage (7–14 days)</li>
        <li>🧠 AI transcription & summarization</li>
        <li>💬 Subject-wise chatbot</li>
        <li>🔐 Student login access</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)


    st.stop()

# ================== LOGIN PAGE ==================
if not st.session_state.logged_in and st.session_state.page == "auth":

    st.markdown('<div class="auth-box">', unsafe_allow_html=True)

    st.markdown("""
    <div class="tabs">
        <div class="tab-active">Login</div>
        <div style="color:#9ca3af;">Signup</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="input-label">User Name</div>', unsafe_allow_html=True)
    username = st.text_input("", key="login_email")

    st.markdown('<div class="input-label">Password</div>', unsafe_allow_html=True)
    password = st.text_input("", type="password", key="login_pass")

    login_btn = st.button("Login")

    st.markdown('</div>', unsafe_allow_html=True)

    # ================== AUTH LOGIC (UNCHANGED) ==================
    if login_btn:
        with open("users.json") as f:
            users = json.load(f)

        if username in users and users[username]["password"] == password:
            st.session_state.logged_in = True
            st.session_state.user = username
            st.session_state.role = users[username]["role"]
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid credentials")

    st.stop()



# ================== SIDEBAR ==================
st.sidebar.markdown("## 👋 TEAM CORE FOUR")
st.sidebar.divider()
st.sidebar.markdown(f"👤 **User:** `{st.session_state.user}`")
st.sidebar.markdown(f"🎭 **Role:** `{st.session_state.role.upper()}`")
st.sidebar.divider()

# ================== UTIL ==================
def clean_text(text):
    return re.sub(r"[^\w\s-]", "", text).replace(" ", "_")

# ================== MENU ==================
menu_items = ["📺 View Lectures", "🤖 AI Chat"]
if st.session_state.role == "staff":
    menu_items.insert(0, "📤 Upload Lecture")

menu = st.sidebar.radio("📌 Menu", menu_items)

# ================== UPLOAD ==================
if menu == "📤 Upload Lecture":
    st.header("📤 Upload Lecture")

    col1, col2 = st.columns(2)
    subject_raw = col1.text_input("Subject Name")
    topic_raw = col2.text_input("Lecture Topic")

    col3, col4 = st.columns(2)
    unit_raw = col3.text_input("Unit / Chapter")
    lecture_date = col4.date_input("Upload Date", value=datetime.now().date())

    col5, col6 = st.columns(2)
    lecture_time = col5.time_input("Upload Time", value=datetime.now().time())

    # 🔹 NEW: Lecture input method
    input_mode = col6.radio(
        "Lecture Input Method",
        ["Upload File", "Record Audio (MP3)"]
    )

    file_bytes = None
    file_ext = None

    # 🔸 Existing upload (unchanged)
    if input_mode == "Upload File":
        file = st.file_uploader("Lecture File", type=["mp4", "mp3", "wav"])
        if file:
            file_bytes = file.read()
            file_ext = file.name.split(".")[-1]

    # 🔸 NEW: Audio recording
    elif input_mode == "Record Audio (MP3)":
        audio = st.audio_input("🎙️ Record Lecture Audio")
        if audio:
            file_bytes = audio.getvalue()
            file_ext = "mp3"


    if st.button("Upload Lecture"):
        if not subject_raw or not unit_raw or not topic_raw or not file_bytes:
            st.error("⚠️ Please fill all fields and provide lecture content")
            st.stop()

        subject = clean_text(subject_raw)
        unit = clean_text(unit_raw)
        topic = clean_text(topic_raw)

        date_str = lecture_date.strftime("%Y-%m-%d")
        time_str = lecture_time.strftime("%H-%M")

        save_dir = os.path.join(BASE_DIR, subject, unit, date_str)
        os.makedirs(save_dir, exist_ok=True)

        filename = f"{subject}_{unit}_{topic}_{time_str}.{file_ext}"
        file_path = os.path.join(save_dir, filename)

        with open(file_path, "wb") as f:
            f.write(file_bytes)

        # 🔹 Transcript placeholder (can be replaced with real STT later)
        transcript_path = file_path.rsplit(".", 1)[0] + ".txt"
        with open(transcript_path, "w") as f:
            f.write(
                f"This lecture covers {topic} from {unit} of {subject}. "
                f"It explains key concepts discussed during the session."
            )

        st.success("✅ Lecture uploaded successfully")

# ================== VIEW ==================
if menu == "📺 View Lectures":
    st.header("📺 Lecture Viewer")

    subjects = os.listdir(BASE_DIR)
    if not subjects:
        st.info("No lectures uploaded yet")
        st.stop()

    subject = st.selectbox("Select Subject", subjects)
    unit = st.selectbox("Select Unit", os.listdir(os.path.join(BASE_DIR, subject)))
    date = st.selectbox(
        "Select Date",
        os.listdir(os.path.join(BASE_DIR, subject, unit))
    )

    date_path = os.path.join(BASE_DIR, subject, unit, date)
    lectures = [f for f in os.listdir(date_path) if f.endswith(("mp4", "mp3", "wav"))]

    if lectures:
        lecture = st.selectbox("Select Lecture", lectures)
        st.session_state.current_path = os.path.join(date_path, lecture)

        if lecture.endswith(".mp4"):
            st.video(st.session_state.current_path)
        else:
            st.audio(st.session_state.current_path)

# ================== AI CHAT (HYBRID KNOWLEDGE) ==================

def is_greeting(text):
    greetings = ["hi", "hai", "hello", "hey", "hii"]
    return text.lower().strip() in greetings


if menu == "🤖 AI Chat":
    st.title("🤖 Classroom AI Chat (Powered by Gemini)")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ---- DISPLAY CHAT HISTORY ----
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "source" in msg:
                st.caption(msg["source"])

    # ---- CHAT INPUT ----
    user_input = st.chat_input("Ask your question...")

    if user_input:
        # ---- USER MESSAGE ----
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        # ---- ASSISTANT RESPONSE ----
        with st.chat_message("assistant"):

            # ✅ STEP 0: Greeting shortcut
            if is_greeting(user_input):
                final_reply = "Hello! 👋 How can I help you with your classroom lectures today?"
                source = "🤖 System Response"

            else:
                lecture_context = load_all_lectures()

                # ✅ STEP 1: Internal lecture-only check
                internal_prompt = f"""
You are Classroom AI.

Answer the question STRICTLY using the classroom lecture content below.
If the answer is not present or insufficient, reply with exactly:
NOT_FOUND

LECTURE CONTENT:
{lecture_context}

QUESTION:
{user_input}
"""

                internal_reply = gemini_chat(internal_prompt)

                # ✅ STEP 2: Decide source
                if (
                    internal_reply.strip() == "NOT_FOUND"
                    or not lecture_context.strip()
                ):
                    # 🔁 External fallback
                    external_prompt = f"""
Answer the following question using your general knowledge.
Explain clearly and in a student-friendly way.

QUESTION:
{user_input}
"""
                    final_reply = gemini_chat(external_prompt)
                    source = "🌐 Source: External Knowledge (Gemini)"
                else:
                    final_reply = internal_reply
                    source = "📘 Source: Classroom Lectures"

            # ---- DISPLAY ASSISTANT ----
            st.markdown(final_reply)
            st.caption(source)

        # ---- SAVE TO SESSION ----
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": final_reply,
                "source": source
            }
        )
# ================== LOGOUT ==================
st.sidebar.divider()
if st.sidebar.button("🚪 Logout"):
    st.session_state.clear()
    st.rerun()
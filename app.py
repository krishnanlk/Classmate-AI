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
from notes_generator import generate_notes_pdf, generate_notes_word
from document_extractor import extract_text_from_document

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Classmate AI",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"Get Help": "https://github.com", "Report a bug": "https://github.com"}
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
/* ========== ROOT CSS VARIABLES ========== */
:root {
    --primary-color: #ff4d4f;
    --secondary-color: #1f2430;
    --bg-dark: #0b0f18;
    --bg-darker: #000000;
    --text-primary: #e5e7eb;
    --text-secondary: #9ca3af;
    --border-color: #374151;
    --input-bg: #2a2f3a;
    --hover-bg: #323846;
    --radius-sm: 8px;
    --radius-md: 14px;
    --radius-lg: 18px;
    --transition: all 0.3s ease;
}

/* ========== GLOBAL RESET & BASE ========== */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

html {
    scroll-behavior: smooth;
}

/* ========== APP CONTAINER ========== */
#MainMenu, footer, header { 
    visibility: hidden; 
}

.stApp {
    background: linear-gradient(135deg, var(--bg-dark) 0%, var(--bg-darker) 100%);
    color: var(--text-primary);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
}

/* ========== BLOCK CONTAINER (RESPONSIVE) ========== */
.block-container {
    padding-top: 2rem;
    max-width: 1400px !important;
    margin: 0 auto !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

/* ========== TYPOGRAPHY ========== */
h1, h2, h3, h4, h5, h6 {
    font-weight: 700;
    line-height: 1.3;
    margin-bottom: 1rem;
}

h1 {
    font-size: clamp(28px, 6vw, 48px);
    color: var(--primary-color);
}

h2 {
    font-size: clamp(24px, 5vw, 36px);
    color: var(--text-primary);
}

h3 {
    font-size: clamp(20px, 4vw, 28px);
    color: var(--text-primary);
}

p {
    font-size: clamp(14px, 2vw, 16px);
    line-height: 1.6;
    color: var(--text-secondary);
    margin-bottom: 0.5rem;
}

/* ========== INPUTS ========== */
input, textarea, select {
    background: var(--input-bg) !important;
    border-radius: var(--radius-sm) !important;
    padding: 12px 14px !important;
    border: 1px solid rgba(255, 77, 79, 0.2) !important;
    color: var(--text-primary) !important;
    font-size: clamp(13px, 2vw, 14px) !important;
    transition: var(--transition) !important;
}

input:hover, textarea:hover, select:hover {
    border-color: var(--primary-color) !important;
    background: var(--hover-bg) !important;
}

input:focus, textarea:focus, select:focus {
    border-color: var(--primary-color) !important;
    box-shadow: 0 0 0 3px rgba(255, 77, 79, 0.1) !important;
}

/* ========== BUTTONS ========== */
.stButton > button {
    background: var(--primary-color) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    font-size: clamp(13px, 2vw, 14px) !important;
    cursor: pointer !important;
    transition: var(--transition) !important;
    min-height: 44px !important;
    display: flex;
    align-items: center;
    justify-content: center;
}

.stButton > button:hover {
    background: #ff6b72 !important;
    transform: translateY(-2px);
    box-shadow: 0 8px 16px rgba(255, 77, 79, 0.3) !important;
}

.stButton > button:active {
    transform: translateY(0);
}

/* ========== SELECTBOX ========== */
div[data-baseweb="select"] > div {
    background-color: var(--input-bg) !important;
    border-radius: var(--radius-sm) !important;
    border: 1px solid rgba(255, 77, 79, 0.2) !important;
    min-height: 44px !important;
}

div[data-baseweb="select"] > div > div {
    background: transparent !important;
}

div[data-baseweb="select"] span {
    color: var(--text-primary) !important;
    font-size: clamp(13px, 2vw, 14px) !important;
}

div[data-baseweb="select"] svg {
    fill: var(--text-secondary) !important;
}

div[data-baseweb="select"]:hover > div {
    background-color: var(--hover-bg) !important;
    border-color: var(--primary-color) !important;
}

/* ========== CHAT MESSAGES ========== */
.stChatMessage {
    background: rgba(255, 77, 79, 0.05) !important;
    border-radius: var(--radius-md) !important;
    padding: 16px !important;
    margin-bottom: 12px !important;
    border-left: 3px solid var(--primary-color) !important;
}

.stChatMessage.user {
    background: rgba(100, 116, 139, 0.1) !important;
    border-left-color: #64748b !important;
}

/* ========== COLUMNS & LAYOUT ========== */
.stColumn {
    padding: 0 8px;
}

/* ========== CARDS & CONTAINERS ========== */
.card-container {
    background: var(--secondary-color);
    border-radius: var(--radius-md);
    padding: 24px;
    border: 1px solid var(--border-color);
    transition: var(--transition);
}

.card-container:hover {
    border-color: var(--primary-color);
    box-shadow: 0 12px 24px rgba(255, 77, 79, 0.1);
}

/* ========== DIVIDER ========== */
hr {
    border: none;
    height: 1px;
    background: var(--border-color);
    margin: 1.5rem 0;
}

/* ========== ALERTS & MESSAGES ========== */
.stSuccess, .stError, .stWarning, .stInfo {
    border-radius: var(--radius-md) !important;
    padding: 16px !important;
    border-left: 4px solid;
}

.stSuccess {
    border-left-color: #10b981 !important;
}

.stError {
    border-left-color: var(--primary-color) !important;
}

.stWarning {
    border-left-color: #f59e0b !important;
}

.stInfo {
    border-left-color: #3b82f6 !important;
}

/* ========== SIDEBAR ========== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--secondary-color) 0%, #141820 100%);
}

section[data-testid="stSidebar"] > div {
    padding: 20px !important;
}

/* ========== RADIO BUTTONS ========== */
div[data-testid="stRadio"] > label {
    font-size: clamp(13px, 2vw, 14px) !important;
}

/* ========== TABS ========== */
.stTabs [data-baseweb="tab-list"] {
    gap: 20px;
    border-bottom: 2px solid var(--border-color);
}

.stTabs [aria-selected="true"] {
    color: var(--primary-color) !important;
    border-bottom: 3px solid var(--primary-color) !important;
}

/* ========== RESPONSIVE: TABLET (768px) ========== */
@media (max-width: 768px) {
    .block-container {
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        padding-top: 1.5rem;
    }

    h1 {
        font-size: 32px;
    }

    h2 {
        font-size: 28px;
    }

    h3 {
        font-size: 24px;
    }

    .stColumn {
        padding: 0 4px;
    }

    .card-container {
        padding: 16px;
    }

    .stButton > button {
        padding: 10px 20px !important;
    }
}

/* ========== RESPONSIVE: MOBILE (480px) ========== */
@media (max-width: 480px) {
    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-top: 1rem;
    }

    h1 {
        font-size: 24px;
        margin-bottom: 0.75rem;
    }

    h2 {
        font-size: 20px;
    }

    h3 {
        font-size: 18px;
    }

    p {
        font-size: 13px;
    }

    input, textarea, select {
        padding: 10px 12px !important;
        font-size: 13px !important;
    }

    .stButton > button {
        padding: 10px 16px !important;
        font-size: 13px !important;
        min-height: 40px !important;
    }

    .stColumn {
        padding: 0;
    }

    .card-container {
        padding: 12px;
    }

    section[data-testid="stSidebar"] > div {
        padding: 12px !important;
    }
}

/* ========== RESPONSIVE: SMALL MOBILE (320px) ========== */
@media (max-width: 320px) {
    .block-container {
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
    }

    h1 {
        font-size: 20px;
    }

    .stButton > button {
        padding: 8px 12px !important;
        font-size: 12px !important;
    }
}

</style>
""", unsafe_allow_html=True)

# ================== SESSION INIT ==================
# Initialize session state first
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user = None
    st.session_state.role = None

# Always restore session from query params if available (for persistence across reloads)
# This ensures session persists even after page refresh
if "user" in st.query_params and "role" in st.query_params:
    # Only restore if not already logged in or if credentials in URL differ
    if not st.session_state.logged_in or st.session_state.user != st.query_params.get("user"):
        st.session_state.logged_in = True
        st.session_state.user = st.query_params.get("user")
        st.session_state.role = st.query_params.get("role")

if "page" not in st.session_state:
    st.session_state.page = "home"

if "page" not in st.query_params:
    st.query_params["page"] = "home"

# Sync page from query_params to session_state
if st.query_params.get("page"):
    st.session_state.page = st.query_params.get("page")

# Page routing logic - only redirect if necessary
logged_in = st.session_state.logged_in
current_page = st.session_state.page

if logged_in:
    # User is logged in
    # Always keep user and role in query params for session persistence
    st.query_params["user"] = st.session_state.user
    st.query_params["role"] = st.session_state.role

    # Redirect from auth/home to dashboard on first login
    if current_page in ["home", "auth"]:
        st.session_state.page = "dashboard"
        st.query_params["page"] = "dashboard"
        st.rerun()
    # Allow dashboard, chat, upload pages
else:
    # User is NOT logged in
    # Clear user and role from query params
    if "user" in st.query_params:
        del st.query_params["user"]
    if "role" in st.query_params:
        del st.query_params["role"]

    # Only allow home and auth pages
    if current_page not in ["home", "auth"]:
        st.session_state.page = "home"
        st.query_params["page"] = "home"
        st.rerun()


# ================== NAVBAR (ALWAYS VISIBLE) ==================
st.markdown("""
<style>
/* ========== BOOTSTRAP-STYLE NAVBAR ========== */
.navbar {
    background: linear-gradient(135deg, #1f2430 0%, #171a22 100%);
    border-bottom: 2px solid rgba(255, 77, 79, 0.2);
    padding: 0;
    margin: 0;
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 999;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
}

.navbar-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 1rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    height: 70px;
    width: 100%;
    box-sizing: border-box;
}

.navbar-brand {
    font-size: 20px;
    font-weight: 700;
    color: #ff4d4f;
    text-decoration: none;
    display: flex;
    align-items: center;
    gap: 8px;
    transition: all 0.3s ease;
    white-space: nowrap;
    cursor: pointer;
}

.navbar-brand:hover {
    transform: scale(1.05);
}

.navbar-menu {
    display: flex;
    gap: 8px;
    list-style: none !important;
    margin: 0 !important;
    padding: 0 !important;
    align-items: center;
}

.navbar-item {
    margin: 0 !important;
    padding: 0 !important;
    list-style: none !important;
}

.navbar-link {
    color: #e5e7eb;
    text-decoration: none;
    padding: 10px 18px;
    border-radius: 8px;
    font-weight: 500;
    font-size: 14px;
    transition: all 0.3s ease;
    display: inline-block;
    white-space: nowrap;
    border: none;
    background: none;
    cursor: pointer;
}

.navbar-link:hover {
    background: #ff4d4f;
    color: white;
    transform: translateY(-2px);
}

.navbar-link.active {
    background: #ff4d4f;
    color: white;
}

.navbar-link.btn-logout {
    background: #ff4d4f;
    color: white;
}

.navbar-link.btn-logout:hover {
    background: #ff6b72;
}

.navbar-toggle {
    display: none;
    background: none !important;
    border: none !important;
    cursor: pointer;
    padding: 8px;
    z-index: 1000;
}

.navbar-toggle span {
    display: block;
    width: 24px;
    height: 2.5px;
    background: #ff4d4f;
    margin: 5px 0;
    border-radius: 2px;
    transition: all 0.3s ease;
}

.navbar-user-info {
    color: #e5e7eb;
    font-size: 13px;
    font-weight: 500;
    margin-right: 12px;
}

/* Checkbox toggle for mobile menu */
#navbar-toggle-checkbox {
    display: none;
}

.navbar-toggle-label {
    display: none;
    flex-direction: column;
    cursor: pointer;
    z-index: 1001;
    gap: 5px;
}

.navbar-toggle-label span {
    width: 24px;
    height: 2.5px;
    background: #ff4d4f;
    border-radius: 2px;
    transition: all 0.3s ease;
}

#navbar-toggle-checkbox:checked ~ .navbar-toggle-label span:nth-child(1) {
    transform: rotate(45deg) translate(8px, 8px);
}

#navbar-toggle-checkbox:checked ~ .navbar-toggle-label span:nth-child(2) {
    opacity: 0;
}

#navbar-toggle-checkbox:checked ~ .navbar-toggle-label span:nth-child(3) {
    transform: rotate(-45deg) translate(7px, -7px);
}

body {
    padding-top: 70px;
}

/* ========== RESPONSIVE: TABLET (768px) ========== */
@media (max-width: 768px) {
    .navbar-container {
        height: 60px;
        padding: 0 0.75rem;
        position: relative;
    }

    .navbar-brand {
        font-size: 18px;
    }

    .navbar-toggle-label {
        display: flex;
    }

    .navbar-menu {
        position: fixed;
        top: 60px;
        left: 0;
        right: 0;
        background: linear-gradient(135deg, #1f2430 0%, #171a22 100%);
        flex-direction: column;
        gap: 0;
        border-bottom: 1px solid rgba(255, 77, 79, 0.2);
        max-height: 0;
        overflow: hidden;
        transition: max-height 0.3s ease;
    }

    #navbar-toggle-checkbox:checked ~ .navbar-menu {
        max-height: 400px;
    }

    .navbar-item {
        width: 100%;
    }

    .navbar-link {
        display: block;
        padding: 14px 1rem;
        border-radius: 0;
        border-left: 4px solid transparent;
        transition: all 0.3s ease;
    }

    .navbar-link:hover {
        border-left-color: #ff4d4f;
        padding-left: 1.5rem;
        transform: none;
    }

    body {
        padding-top: 60px;
    }
}

/* ========== RESPONSIVE: MOBILE (480px) ========== */
@media (max-width: 480px) {
    .navbar-container {
        height: 56px;
        padding: 0 0.5rem;
    }

    .navbar-brand {
        font-size: 16px;
        gap: 4px;
    }

    .navbar-link {
        padding: 12px 0.75rem;
        font-size: 13px;
    }

    #navbar-toggle-checkbox:checked ~ .navbar-menu {
        max-height: 380px;
    }

    .navbar-user-info {
        display: none;
    }

    body {
        padding-top: 56px;
    }
}
</style>
""", unsafe_allow_html=True)

# Check for logout action from navbar
if st.query_params.get("action") == "logout":
    # Clear session state
    st.session_state.logged_in = False
    st.session_state.user = None
    st.session_state.role = None
    st.session_state.page = "home"

    # Clear all query params
    st.query_params.clear()
    st.query_params["page"] = "home"
    st.rerun()

# Navbar content based on login state
if st.session_state.logged_in:
    # Include user and role in all navigation links to preserve session
    user_param = st.session_state.user
    role_param = st.session_state.role
    upload_link = f'<li class="navbar-item"><a href="?page=upload&user={user_param}&role={role_param}" class="navbar-link" target="_self">📤 Upload</a></li>' if st.session_state.role == "staff" else ''
    navbar_html = f"""<nav class="navbar"><div class="navbar-container"><a href="?page=dashboard&user={user_param}&role={role_param}" class="navbar-brand" target="_self">🎓 Classmate AI</a><input type="checkbox" id="navbar-toggle-checkbox"><label for="navbar-toggle-checkbox" class="navbar-toggle-label"><span></span><span></span><span></span></label><ul class="navbar-menu"><li class="navbar-item"><a href="?page=dashboard&user={user_param}&role={role_param}" class="navbar-link" target="_self">📺 Lectures</a></li><li class="navbar-item"><a href="?page=chat&user={user_param}&role={role_param}" class="navbar-link" target="_self">🤖 Chat</a></li>{upload_link}<li class="navbar-item"><span class="navbar-user-info">👤 {st.session_state.user}</span></li><li class="navbar-item"><a href="?action=logout" class="navbar-link btn-logout" target="_self">🚪 Logout</a></li></ul></div></nav>"""
    st.markdown(navbar_html, unsafe_allow_html=True)
else:
    navbar_html = """<nav class="navbar"><div class="navbar-container"><a href="?page=home" class="navbar-brand" target="_self">🎓 Classmate AI</a><input type="checkbox" id="navbar-toggle-checkbox"><label for="navbar-toggle-checkbox" class="navbar-toggle-label"><span></span><span></span><span></span></label><ul class="navbar-menu"><li class="navbar-item"><a href="?page=home" class="navbar-link" target="_self">🏠 Home</a></li><li class="navbar-item"><a href="?page=auth" class="navbar-link active" target="_self">🔐 Login</a></li></ul></div></nav>"""
    st.markdown(navbar_html, unsafe_allow_html=True)

# ================== HOME PAGE ==================
if not st.session_state.logged_in and st.session_state.page == "home":

    st.markdown("""
    <style>
    .hero-section {
        background: linear-gradient(135deg, rgba(255, 77, 79, 0.1) 0%, rgba(31, 36, 48, 0.5) 100%);
        border: 1px solid rgba(255, 77, 79, 0.2);
        border-radius: 18px;
        padding: clamp(24px, 5vw, 48px);
        margin: 2rem 0;
        text-align: center;
        animation: slideIn 0.6s ease;
    }

    .hero-section h1 {
        background: linear-gradient(135deg, #ff4d4f 0%, #ff6b72 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
    }

    .features-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 20px;
        margin-top: 2rem;
    }

    .feature-card {
        background: linear-gradient(135deg, #1f2430 0%, #171a22 100%);
        border: 1px solid rgba(255, 77, 79, 0.15);
        border-radius: 14px;
        padding: 24px;
        transition: all 0.3s ease;
        cursor: default;
    }

    .feature-card:hover {
        border-color: #ff4d4f;
        transform: translateY(-8px);
        box-shadow: 0 12px 32px rgba(255, 77, 79, 0.15);
    }

    .feature-card h3 {
        color: #ff4d4f;
        margin-top: 12px;
        margin-bottom: 12px;
    }

    .feature-card p {
        color: #9ca3af;
        font-size: 14px;
        line-height: 1.6;
    }

    .feature-icon {
        font-size: 2.5rem;
        line-height: 1;
    }

    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @media (max-width: 768px) {
        .hero-section {
            padding: 24px;
        }

        .features-grid {
            grid-template-columns: 1fr;
            gap: 16px;
        }

        .feature-card {
            padding: 18px;
        }
    }

    @media (max-width: 480px) {
        .hero-section {
            padding: 16px;
            margin: 1rem 0;
        }

        .feature-card {
            padding: 14px;
        }

        .feature-icon {
            font-size: 2rem;
        }
    }
    </style>

    <div class="hero-section">
        <h1>🎓 Welcome to Classmate AI</h1>
        <p style="font-size: clamp(14px, 3vw, 18px); color: #cbd5e1; margin-bottom: 0;">
            Your intelligent AI-powered classroom assistant designed to capture,
            organize, and enhance your learning experience.
        </p>
    </div>

    <div class="features-grid">
        <div class="feature-card">
            <div class="feature-icon">🎤</div>
            <h3>Auto Lecture Recording</h3>
            <p>Automatically Record audio and store lectures for future reference with crystal-clear quality.</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">☁️</div>
            <h3>Secure Storage</h3>
            <p>Your lectures are safely stored in the local server for No.of.Days days with encrypted access.</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🧠</div>
            <h3>AI Transcription & Summarization</h3>
            <p>Get automatic transcripts and summaries to quickly understand key concepts.</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">💬</div>
            <h3>Personal Chatbot</h3>
            <p>Ask questions and get instant answers directly from your classroom content with the help of Gemini AI.</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🔐</div>
            <h3>Student Login Access</h3>
            <p>Secure authentication ensures only authorized students can access lectures.</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <h3>Lightning Fast Search</h3>
            <p>Find any lecture topic in seconds with our intelligent search engine.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.stop()
## ======================login page===================##
if not st.session_state.logged_in and st.session_state.page == "auth":

    st.markdown("""
    <style>
    header, footer, #MainMenu {display:none;}

    [data-testid="stAppViewContainer"] > .main {
        min-height: 100vh;
        background: radial-gradient(circle at top, #0b0f19, #02040a);
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 0;
        .block-container {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        }

    }

    .block-container {
        max-width: 1100px !important;
        padding: 0 !important;
    }

    /* RED WRAPPER */
    .block-container {
        max-width: 720px !important;
        padding: 36px !important;
        background: linear-gradient(135deg, #0f172a, #020617);
        border-radius: 28px;
        box-shadow: 0 40px 120px rgba(255,77,79,0.6);
    }

    .inner-card {
        background: #0b0f19;
        border-radius: 22px;
        padding: 40px 48px;
        color: white;
    }

    }

    /* DARK INNER CARD */
    .inner-card {
        background: #0b0f19;
        border-radius: 22px;
        padding: 40px 48px;
        color: white;
    }

    .title {
        text-align: center;
        margin-bottom: 30px;
    }

    .title h1 {
        color: #ff4d4f;
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 6px;
    }

    .title p {
        color: #cbd5e1;
        font-size: 14px;
    }

    div[data-testid="stTextInput"] input {
        height: 50px;
        background: #1f2430;
        border-radius: 12px;
        border: 2px solid rgba(255,77,79,0.35);
        color: white;
    }

    div[data-testid="stButton"] button {
        height: 52px;
        font-weight: 700;
        border-radius: 14px;
        background: linear-gradient(135deg, #0f172a, #020617);
        box-shadow: 0 10px 30px rgba(255,77,79,0.5);
    }

    .demo {
        margin-top: 28px;
        padding: 18px;
        border-radius: 14px;
        background: rgba(59,130,246,0.12);
        border: 1px solid rgba(59,130,246,0.3);
        text-align: center;
        font-size: 13px;
        color: #cbd5e1;
    }
    </style>
    """, unsafe_allow_html=True)

    # ===== STRUCTURE THAT ACTUALLY WORKS =====
    with st.container():
        st.markdown('<div class="red-shell"><div class="inner-card">', unsafe_allow_html=True)

        st.markdown("""
        <div class="title">
            <h1>🔐 Login</h1>
            <p>Access your classroom lectures and AI-powered learning</p>
        </div>
        """, unsafe_allow_html=True)

        username = st.text_input("👤 Username", placeholder="Enter your username")
        password = st.text_input("🔑 Password", type="password", placeholder="Enter your password")

        login = st.button("🚀 Login to Dashboard", use_container_width=True)

        st.markdown("""
        <div class="demo">
            <b>Demo Accounts</b><br><br>
            Student: <b>stu1</b> / <b>stu123</b><br>
            Staff: <b>staff1</b> / <b>staff123</b>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('</div></div>', unsafe_allow_html=True)

        if login:
            with open("users.json") as f:
                users = json.load(f)

            if username in users and users[username]["password"] == password:
                st.session_state.logged_in = True
                st.session_state.user = username
                st.session_state.role = users[username]["role"]
                st.query_params["user"] = username
                st.query_params["role"] = users[username]["role"]
                st.query_params["page"] = "dashboard"
                st.rerun()
            else:
                st.error("❌ Invalid credentials")

    st.stop()
# ================== SIDEBAR ==================
st.sidebar.markdown("""
<style>
.sidebar-header {
    color: #ff4d4f;
    font-size: 16px;
    font-weight: 700;
    margin-bottom: 1rem;
}

.sidebar-item {
    padding: 10px;
    color: #e5e7eb;
    font-size: 14px;
    margin: 8px 0;
}

.sidebar-divider {
    border-top: 1px solid rgba(255, 77, 79, 0.2);
    margin: 1rem 0;
}

/* Sidebar radio button active state styling */
div[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: transparent !important;
    padding: 12px 16px !important;
    border-radius: 8px !important;
    margin-bottom: 8px !important;
    transition: all 0.3s ease !important;
    border: 1px solid transparent !important;
}

div[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: rgba(255, 77, 79, 0.1) !important;
    border-color: rgba(255, 77, 79, 0.3) !important;
}

div[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] {
    background: linear-gradient(135deg, rgba(255, 77, 79, 0.2) 0%, rgba(255, 77, 79, 0.1) 100%) !important;
    border-color: #ff4d4f !important;
    border-left: 4px solid #ff4d4f !important;
    padding-left: 12px !important;
}

div[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] span {
    color: #ff4d4f !important;
    font-weight: 600 !important;
}

div[data-testid="stSidebar"] div[role="radiogroup"] label span {
    font-size: 14px !important;
    color: #e5e7eb !important;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("<div class='sidebar-header'>👋 TEAM CORE FOUR</div>", unsafe_allow_html=True)
st.sidebar.divider()
st.sidebar.markdown(f"<div class='sidebar-item'>👤 <strong>User:</strong> {st.session_state.user if st.session_state.user else 'Not logged in'}</div>", unsafe_allow_html=True)
st.sidebar.markdown(f"<div class='sidebar-item'>🎭 <strong>Role:</strong> {st.session_state.role if st.session_state.role else 'None'}</div>", unsafe_allow_html=True)
st.sidebar.divider()

# Only show menu and pages if logged in
if not st.session_state.logged_in:
    st.stop()

# ================== UTIL ==================
def clean_text(text):
    return re.sub(r"[^\w\s-]", "", text).replace(" ", "_")

# ================== NOTES GENERATION ==================
def generate_key_notes(lecture_title, lecture_subject, lecture_transcript):
    """
    Generate key notes from lecture transcript using Gemini AI.
    
    Args:
        lecture_title (str): Title of the lecture
        lecture_subject (str): Subject name
        lecture_transcript (str): The lecture transcript/content
    
    Returns:
        str: Key notes formatted for PDF/Word export
    """
    if not lecture_transcript or len(lecture_transcript.strip()) < 100:
        return "No sufficient lecture content available to generate notes."
    
    prompt = f"""
You are an expert note-taking assistant. Extract the KEY IMPORTANT POINTS from the following lecture content.

LECTURE SUBJECT: {lecture_subject}
LECTURE TITLE: {lecture_title}

LECTURE CONTENT:
{lecture_transcript}

Please provide:
1. A brief summary (2-3 sentences)
2. Key concepts and definitions (as bullet points)
3. Important formulas or equations (if any)
4. Key takeaways (main points to remember)

Format the output clearly with headers and bullet points. Make it concise but comprehensive.
"""
    
    notes = gemini_chat(prompt)
    return notes

# ================== CHAT HISTORY MANAGEMENT ==================
def get_chat_history_path(user_id):
    """Get the directory path for storing user chat histories."""
    chat_history_dir = os.path.join(CHAT_DIR, user_id, "conversations")
    os.makedirs(chat_history_dir, exist_ok=True)
    return chat_history_dir

def save_chat_conversation(user_id, conversation_id, messages, title=None):
    """
    Save a conversation to a JSON file.
    
    Args:
        user_id: User identifier
        conversation_id: Unique ID for the conversation
        messages: List of message dictionaries
        title: Optional title for the conversation
    """
    chat_dir = get_chat_history_path(user_id)
    conversation_file = os.path.join(chat_dir, f"{conversation_id}.json")
    
    # Create default title from first message if not provided
    if not title and messages:
        first_msg = messages[0].get("content", "Untitled")[:50]
        title = first_msg
    
    conversation_data = {
        "id": conversation_id,
        "title": title or "Untitled Conversation",
        "created_at": datetime.now().isoformat(),
        "last_modified": datetime.now().isoformat(),
        "messages": messages
    }
    
    with open(conversation_file, "w", encoding="utf-8") as f:
        json.dump(conversation_data, f, indent=2, ensure_ascii=False)

def load_chat_conversation(user_id, conversation_id):
    """
    Load a conversation from a JSON file.
    
    Args:
        user_id: User identifier
        conversation_id: Unique ID for the conversation
    
    Returns:
        dict: Conversation data or None if not found
    """
    chat_dir = get_chat_history_path(user_id)
    conversation_file = os.path.join(chat_dir, f"{conversation_id}.json")
    
    if os.path.exists(conversation_file):
        with open(conversation_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def list_chat_conversations(user_id):
    """
    List all conversations for a user.
    
    Args:
        user_id: User identifier
    
    Returns:
        list: List of conversation metadata sorted by most recent first
    """
    chat_dir = get_chat_history_path(user_id)
    conversations = []
    
    if os.path.exists(chat_dir):
        for filename in os.listdir(chat_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(chat_dir, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    conversations.append({
                        "id": data.get("id"),
                        "title": data.get("title", "Untitled"),
                        "created_at": data.get("created_at"),
                        "last_modified": data.get("last_modified", data.get("created_at"))
                    })
    
    # Sort by last_modified descending (most recent first)
    conversations.sort(key=lambda x: x.get("last_modified", ""), reverse=True)
    return conversations

def delete_chat_conversation(user_id, conversation_id):
    """
    Delete a conversation.
    
    Args:
        user_id: User identifier
        conversation_id: Unique ID for the conversation
    """
    chat_dir = get_chat_history_path(user_id)
    conversation_file = os.path.join(chat_dir, f"{conversation_id}.json")
    
    if os.path.exists(conversation_file):
        os.remove(conversation_file)

def generate_conversation_id():
    """Generate a unique conversation ID based on timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]

# ================== INITIALIZE CHAT SESSION ==================
if "page" in st.session_state and st.session_state.page == "chat":
    # Initialize chat session state variables if not already done
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "document_context" not in st.session_state:
        st.session_state.document_context = None
    if "document_name" not in st.session_state:
        st.session_state.document_name = None

# ================== MENU ==================
menu_items = ["📺 View Lectures", "🤖 AI Chat"]
if st.session_state.role == "staff":
    menu_items.insert(0, "📤 Upload Lecture")

# Map page query param to menu items
page_to_menu = {
    "dashboard": "📺 View Lectures",
    "chat": "🤖 AI Chat",
    "upload": "📤 Upload Lecture"
}

# Determine which menu item to select
default_menu_index = 0
if st.session_state.page in page_to_menu:
    menu_label = page_to_menu[st.session_state.page]
    if menu_label in menu_items:
        default_menu_index = menu_items.index(menu_label)

menu = st.sidebar.radio("📌 Menu", menu_items, index=default_menu_index)

# Update page based on menu selection
menu_to_page = {
    "📺 View Lectures": "dashboard",
    "🤖 AI Chat": "chat",
    "📤 Upload Lecture": "upload"
}
if menu in menu_to_page:
    st.session_state.page = menu_to_page[menu]
    st.query_params["page"] = menu_to_page[menu]

# ================== UPLOAD ==================
if menu == "📤 Upload Lecture":
    st.markdown("""
    <style>
    .upload-section {
        background: linear-gradient(135deg, #1f2430 0%, #171a22 100%);
        border: 1px solid rgba(255, 77, 79, 0.2);
        border-radius: 16px;
        padding: clamp(20px, 5vw, 32px);
        margin-bottom: 2rem;
    }

    .form-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 16px;
        margin-bottom: 1.5rem;
    }

    @media (max-width: 768px) {
        .form-grid {
            grid-template-columns: 1fr;
            gap: 12px;
        }
    }
    </style>

    <div class="upload-section">
    <h2 style="color: #ff4d4f; margin-bottom: 1.5rem;">📤 Upload Lecture</h2>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        subject_raw = st.text_input("📚 Subject Name", placeholder="e.g., AI, DAA, DBMS")
    with col2:
        topic_raw = st.text_input("📝 Lecture Topic", placeholder="e.g., Machine Learning Basics")

    col3, col4 = st.columns([1, 1])
    with col3:
        unit_raw = st.text_input("📖 Unit / Chapter", placeholder="e.g., Unit 1, Chapter 3")
    with col4:
        lecture_date = st.date_input("📅 Upload Date", value=datetime.now().date())

    col5, col6 = st.columns([1, 1])
    with col5:
        lecture_time = st.time_input("⏰ Upload Time", value=datetime.now().time())
    with col6:
        input_mode = st.radio("Input Method", ["Upload File", "Record Audio (MP3)"], horizontal=True)

    file_bytes = None
    file_ext = None

    if input_mode == "Upload File":
        file = st.file_uploader("🎬 Select Lecture File", type=["mp4", "mp3", "wav"])
        if file:
            file_bytes = file.read()
            file_ext = file.name.split(".")[-1]

    elif input_mode == "Record Audio (MP3)":
        audio = st.audio_input("🎙️ Record Lecture Audio")
        if audio:
            file_bytes = audio.getvalue()
            file_ext = "mp3"

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🚀 Upload Lecture", use_container_width=True):
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

            transcript_path = file_path.rsplit(".", 1)[0] + ".txt"
            with open(transcript_path, "w") as f:
                f.write(
                    f"This lecture covers {topic} from {unit} of {subject}. "
                    f"It explains key concepts discussed during the session."
                )

            st.success("✅ Lecture uploaded successfully!")
            st.balloons()

# ================== VIEW ==================
if menu == "📺 View Lectures":
    # Add dashboard styling
    st.markdown("""
    <style>
    /* Dashboard header */
    .dashboard-header {
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, rgba(255, 77, 79, 0.1) 0%, rgba(31, 36, 48, 0.5) 100%);
        border-radius: 16px;
        border: 1px solid rgba(255, 77, 79, 0.2);
    }

    .dashboard-header h1 {
        color: #ff4d4f;
        font-size: clamp(24px, 5vw, 36px);
        margin-bottom: 0.5rem;
        font-weight: 700;
    }

    /* Selectbox styling improvements */
    div[data-baseweb="select"] {
        margin-bottom: 1.5rem !important;
    }

    /* Video/Audio player container */
    .element-container iframe,
    .element-container video,
    .element-container audio {
        border-radius: 12px !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3) !important;
        margin-top: 1.5rem !important;
    }

    /* Info messages */
    .stInfo {
        background: rgba(59, 130, 246, 0.1) !important;
        border-left-color: #3b82f6 !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Dashboard header
    st.markdown("""
    <div class="dashboard-header">
        <h1>📺 Lecture Viewer</h1>
        <p style="color: #9ca3af; font-size: clamp(14px, 2vw, 16px);">Browse and watch your classroom lectures</p>
    </div>
    """, unsafe_allow_html=True)

    subjects = os.listdir(BASE_DIR)
    if not subjects:
        st.info("📚 No lectures uploaded yet. Staff can upload lectures from the Upload page.")
        st.stop()

    # Selectboxes in columns for better layout
    col1, col2 = st.columns(2)
    with col1:
        subject = st.selectbox("📚 Select Subject", subjects)

    with col2:
        unit = st.selectbox("📖 Select Unit", os.listdir(os.path.join(BASE_DIR, subject)))

    date = st.selectbox(
        "📅 Select Date",
        os.listdir(os.path.join(BASE_DIR, subject, unit))
    )

    date_path = os.path.join(BASE_DIR, subject, unit, date)
    lectures = [f for f in os.listdir(date_path) if f.endswith(("mp4", "mp3", "wav"))]

    if lectures:
        lecture = st.selectbox("🎬 Select Lecture", lectures)
        st.session_state.current_path = os.path.join(date_path, lecture)

        st.divider()

        if lecture.endswith(".mp4"):
            st.video(st.session_state.current_path)
        else:
            st.audio(st.session_state.current_path)
        
        # ================== NOTES DOWNLOAD SECTION ==================
        st.markdown("""
        <style>
        .notes-section {
            background: linear-gradient(135deg, rgba(255, 77, 79, 0.08) 0%, rgba(31, 36, 48, 0.4) 100%);
            border: 1px solid rgba(255, 77, 79, 0.2);
            border-radius: 12px;
            padding: 1.5rem;
            margin-top: 2rem;
        }
        
        .notes-section h3 {
            color: #ff4d4f;
            margin-top: 0;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .notes-buttons {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            margin-top: 1rem;
        }
        
        @media (max-width: 768px) {
            .notes-buttons {
                flex-direction: column;
            }
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="notes-section">
            <h3>📝 Download Lecture Notes</h3>
            <p style="color: #9ca3af; margin-bottom: 1rem; font-size: 14px;">
                Generate and download important notes from this lecture as PDF or Word document
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col_notes1, col_notes2 = st.columns(2)
        
        with col_notes1:
            if st.button("📄 Download as PDF", use_container_width=True, key="download_pdf"):
                with st.spinner("⏳ Generating PDF notes..."):
                    try:
                        # Load lecture transcript
                        transcript_path = st.session_state.current_path.rsplit(".", 1)[0] + ".txt"
                        if os.path.exists(transcript_path):
                            with open(transcript_path, "r", encoding="utf-8") as f:
                                transcript = f.read()
                        else:
                            transcript = "Lecture content not available. Please check the transcript file."
                        
                        # Generate key notes using Gemini AI
                        key_notes = generate_key_notes(
                            lecture_title=lecture.replace(".mp4", "").replace(".mp3", "").replace(".wav", ""),
                            lecture_subject=subject,
                            lecture_transcript=transcript
                        )
                        
                        # Generate PDF
                        pdf_content = generate_notes_pdf(
                            lecture_title=lecture.replace(".mp4", "").replace(".mp3", "").replace(".wav", ""),
                            lecture_subject=subject,
                            lecture_notes=key_notes,
                            lecture_date=date
                        )
                        
                        # Create download button
                        filename = f"{subject}_{lecture.rsplit('.', 1)[0]}_notes.pdf"
                        st.download_button(
                            label="✅ Click to download PDF",
                            data=pdf_content,
                            file_name=filename,
                            mime="application/pdf",
                            use_container_width=True
                        )
                        st.success("✅ PDF notes generated successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ Error generating PDF: {str(e)}")
        
        with col_notes2:
            if st.button("📋 Download as Word", use_container_width=True, key="download_word"):
                with st.spinner("⏳ Generating Word notes..."):
                    try:
                        # Load lecture transcript
                        transcript_path = st.session_state.current_path.rsplit(".", 1)[0] + ".txt"
                        if os.path.exists(transcript_path):
                            with open(transcript_path, "r", encoding="utf-8") as f:
                                transcript = f.read()
                        else:
                            transcript = "Lecture content not available. Please check the transcript file."
                        
                        # Generate key notes using Gemini AI
                        key_notes = generate_key_notes(
                            lecture_title=lecture.replace(".mp4", "").replace(".mp3", "").replace(".wav", ""),
                            lecture_subject=subject,
                            lecture_transcript=transcript
                        )
                        
                        # Generate Word document
                        word_content = generate_notes_word(
                            lecture_title=lecture.replace(".mp4", "").replace(".mp3", "").replace(".wav", ""),
                            lecture_subject=subject,
                            lecture_notes=key_notes,
                            lecture_date=date
                        )
                        
                        # Create download button
                        filename = f"{subject}_{lecture.rsplit('.', 1)[0]}_notes.docx"
                        st.download_button(
                            label="✅ Click to download Word",
                            data=word_content,
                            file_name=filename,
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            use_container_width=True
                        )
                        st.success("✅ Word notes generated successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ Error generating Word document: {str(e)}")

# ================== AI CHAT (HYBRID KNOWLEDGE) ==================

def is_greeting(text):
    greetings = ["hi", "hai", "hello", "hey", "hii","greetings", "good morning", "good afternoon", "good evening"]
    return text.lower().strip() in greetings


if menu == "🤖 AI Chat":
    # Initialize chat session state variables
    if "current_conversation_id" not in st.session_state:
        st.session_state.current_conversation_id = generate_conversation_id()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "document_context" not in st.session_state:
        st.session_state.document_context = None
    if "document_name" not in st.session_state:
        st.session_state.document_name = None
    
    # Add comprehensive chat page styling
    st.markdown("""
    <style>
    /* Chat page specific styling */
    .chat-header {
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, rgba(255, 77, 79, 0.1) 0%, rgba(31, 36, 48, 0.5) 100%);
        border-radius: 16px;
        border: 1px solid rgba(255, 77, 79, 0.2);
    }

    .chat-header h1 {
        color: #ff4d4f;
        font-size: clamp(24px, 5vw, 36px);
        margin-bottom: 0.5rem;
        font-weight: 700;
    }

    .chat-header p {
        color: #9ca3af;
        font-size: clamp(14px, 2vw, 16px);
        margin: 0;
    }

    /* Chat messages styling */
    .stChatMessage {
        background: rgba(31, 36, 48, 0.6) !important;
        border-radius: 12px !important;
        padding: 16px !important;
        margin-bottom: 16px !important;
        border: 1px solid rgba(255, 77, 79, 0.1) !important;
    }

    /* User messages */
    .stChatMessage[data-testid="user-message"] {
        background: rgba(100, 116, 139, 0.2) !important;
        border-left: 3px solid #64748b !important;
    }

    /* Assistant messages */
    .stChatMessage[data-testid="assistant-message"] {
        background: rgba(255, 77, 79, 0.08) !important;
        border-left: 3px solid #ff4d4f !important;
    }

    /* Chat input styling */
    .stChatInputContainer {
        background: var(--secondary-color) !important;
        border-radius: 12px !important;
        padding: 8px !important;
        border: 1px solid rgba(255, 77, 79, 0.2) !important;
        margin-top: 2rem !important;
    }

    .stChatInputContainer textarea {
        background: var(--input-bg) !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255, 77, 79, 0.2) !important;
        color: var(--text-primary) !important;
        font-size: 15px !important;
        padding: 12px !important;
        min-height: 50px !important;
    }

    .stChatInputContainer textarea:focus {
        border-color: var(--primary-color) !important;
        box-shadow: 0 0 0 2px rgba(255, 77, 79, 0.1) !important;
    }

    /* Source caption styling */
    .stChatMessage .element-container p {
        line-height: 1.6;
        margin-bottom: 0.5rem;
    }

    /* Empty state message */
    .chat-empty-state {
        text-align: center;
        padding: 4rem 2rem;
        color: #9ca3af;
    }

    .chat-empty-state h3 {
        color: #ff4d4f;
        margin-bottom: 1rem;
    }

    /* File upload button styling */
    .file-upload-btn {
        background: linear-gradient(135deg, rgba(255, 77, 79, 0.2) 0%, rgba(255, 77, 79, 0.1) 100%);
        border: 1px solid rgba(255, 77, 79, 0.3);
        border-radius: 8px;
        padding: 10px 16px;
        color: #ff4d4f;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }

    .file-upload-btn:hover {
        background: linear-gradient(135deg, rgba(255, 77, 79, 0.3) 0%, rgba(255, 77, 79, 0.2) 100%);
        border-color: #ff4d4f;
    }
    
    /* Style file uploader button */
    div[data-testid="stFileUploader"] {
        width: 100%;
    }
    
    div[data-testid="stFileUploader"] > div {
        padding: 0 !important;
    }
    
    div[data-testid="stFileUploader"] button {
        width: 100% !important;
        padding: 8px !important;
        background: rgba(255, 77, 79, 0.1) !important;
        border: 1px solid rgba(255, 77, 79, 0.3) !important;
        border-radius: 8px !important;
        color: #ff4d4f !important;
        font-size: 18px !important;
        transition: all 0.3s ease !important;
    }
    
    div[data-testid="stFileUploader"] button:hover {
        background: rgba(255, 77, 79, 0.2) !important;
        border-color: #ff4d4f !important;
    }

    /* Sidebar history styling */
    .chat-history-item {
        padding: 12px;
        margin-bottom: 8px;
        background: rgba(255, 77, 79, 0.08);
        border-left: 3px solid transparent;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.3s ease;
        word-break: break-word;
    }

    .chat-history-item:hover {
        background: rgba(255, 77, 79, 0.15);
        border-left-color: #ff4d4f;
    }

    .chat-history-item.active {
        background: rgba(255, 77, 79, 0.2);
        border-left-color: #ff4d4f;
    }

    .document-badge {
        display: inline-block;
        background: #ff4d4f;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        margin-bottom: 10px;
    }

    @media (max-width: 768px) {
        .chat-header {
            padding: 1.5rem 1rem;
            margin-bottom: 1.5rem;
        }

        .stChatMessage {
            padding: 12px !important;
            margin-bottom: 12px !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    # ==================== MAIN CHAT AREA ====================
    # Chat header
    st.markdown("""
    <div class="chat-header">
        <h1>🤖 Classroom AI Chat</h1>
        <p>Powered by Gemini - Ask questions about your lectures or general topics</p>
    </div>
    """, unsafe_allow_html=True)

    # Display uploaded document info at top if present
    if st.session_state.document_context:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"""
            <div class="document-badge">
                📎 Uploaded: {st.session_state.document_name}
            </div>
            """, unsafe_allow_html=True)
        with col2:
            if st.button("❌ Remove", use_container_width=True, key="remove_doc"):
                st.session_state.document_context = None
                st.session_state.document_name = None
                st.rerun()

    # Show empty state if no messages
    if len(st.session_state.messages) == 0:
        st.markdown("""
        <div class="chat-empty-state">
            <h3>💬 Start a Conversation</h3>
            <p>Ask me anything about your classroom lectures or general knowledge!</p>
            <p style="margin-top: 1rem; font-size: 14px;">
                💡 Tip: Upload a PDF or Word document (📎) to ask questions about its content!
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ---- DISPLAY CHAT HISTORY ----
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "source" in msg:
                st.caption(msg["source"])

    # ---- CHAT INPUT WITH FILE UPLOAD BUTTON ----
    col_btn, col_input = st.columns([1.2, 9.2])
    
    with col_btn:
        # File uploader button
        uploaded_file = st.file_uploader(
            "Attach File",
            type=["pdf", "docx", "doc"],
            label_visibility="visible",
            key="file_upload"
        )
        
        if uploaded_file:
            with st.spinner("Processing document..."):
                try:
                    file_ext = uploaded_file.name.split(".")[-1].lower()
                    extracted_text, filename = extract_text_from_document(uploaded_file, file_ext)
                    st.session_state.document_context = extracted_text
                    st.session_state.document_name = filename
                    st.success(f"✅ {filename} uploaded!")
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
    
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
                # Build context with document + lectures
                document_context = st.session_state.document_context or ""
                lecture_context = load_all_lectures()
                
                combined_context = ""
                source = "🌐 Source: External Knowledge (Gemini)"
                
                if st.session_state.document_name and document_context:
                    combined_context = f"UPLOADED DOCUMENT: {st.session_state.document_name}\n{document_context}\n\n"
                    source = f"📄 Source: {st.session_state.document_name}"
                
                combined_context += f"CLASSROOM LECTURES:\n{lecture_context}"

                # ✅ STEP 1: Internal check with document + lectures
                internal_prompt = f"""
You are Classroom AI.

Answer the question STRICTLY using the content provided below.
If the answer is not present or insufficient in the provided content, you may use general knowledge.

AVAILABLE CONTENT:
{combined_context}

QUESTION:
{user_input}
"""

                internal_reply = gemini_chat(internal_prompt)

                # Check if answer was found in documents
                if document_context and st.session_state.document_name:
                    final_reply = internal_reply
                    source = f"📄 Source: {st.session_state.document_name}"
                elif lecture_context.strip():
                    final_reply = internal_reply
                    source = "📘 Source: Classroom Lectures"
                else:
                    final_reply = internal_reply
                    source = "🌐 Source: General Knowledge (Gemini)"

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
        
        # Auto-save conversation every time a message is sent
        save_chat_conversation(st.session_state.user, st.session_state.current_conversation_id, st.session_state.messages)

# ================== LOGOUT ==================
st.sidebar.divider()
if st.sidebar.button("🚪 Logout"):
    # Clear all session state variables
    for key in list(st.session_state.keys()):
        del st.session_state[key]

    # Re-initialize session state
    st.session_state.logged_in = False
    st.session_state.user = None
    st.session_state.role = None
    st.session_state.page = "home"

    # Clear all query params
    st.query_params.clear()
    st.query_params["page"] = "home"

    # Rerun to show login page
    st.rerun()
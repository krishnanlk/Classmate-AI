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
    --bg-dark: #0b0f19;
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
            <div class="feature-icon">🎥</div>
            <h3>Auto Lecture Recording</h3>
            <p>Automatically capture and store lectures for future reference with crystal-clear quality.</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">☁️</div>
            <h3>Secure Cloud Storage</h3>
            <p>Your lectures are safely stored in the cloud for 7–14 days with encrypted access.</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🧠</div>
            <h3>AI Transcription & Summarization</h3>
            <p>Get automatic transcripts and summaries to quickly understand key concepts.</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">💬</div>
            <h3>Subject-wise Chatbot</h3>
            <p>Ask questions and get instant answers from your classroom content.</p>
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

# ================== LOGIN PAGE ==================
if not st.session_state.logged_in and st.session_state.page == "auth":

    st.markdown("""
    <style>
    /* Hide all Streamlit default elements on login page */
    [data-testid="stHeader"],
    [data-testid="stDecoration"],
    [data-testid="stToolbar"],
    .stDeployButton,
    #MainMenu,
    footer {
        display: none !important;
    }

    /* Full screen background */
    [data-testid="stAppViewContainer"] > .main {
        background: linear-gradient(135deg, #0b0f19 0%, #1a1f2e 50%, #0b0f19 100%) !important;
        padding: 0 !important;
    }

    .block-container {
        display: flex !important;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        min-height: 100vh !important;
        padding: 2rem 1rem !important;
        max-width: 100% !important;
    }

    /* Hide all extra element containers */
    .element-container {
        width: 100%;
        max-width: 100%;
    }

    .element-container:first-child:not(:has(div)) {
        display: none !important;
    }

    /* Ensure login wrapper takes proper width */
    .element-container:has(.login-wrapper) {
        width: 100%;
        max-width: 100%;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    /* Login wrapper */
    .login-wrapper {
        max-width: 500px;
        width: 100%;
        margin: 0 auto;
        padding: 0;
    }

    /* Login card with glassmorphism */
    .login-card {
        background: rgba(31, 36, 48, 0.9);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 77, 79, 0.3);
        border-radius: 24px;
        padding: 3rem 2.5rem;
        box-shadow:
            0 20px 60px rgba(0, 0, 0, 0.7),
            0 0 100px rgba(255, 77, 79, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        animation: slideUp 0.7s cubic-bezier(0.16, 1, 0.3, 1);
        position: relative;
        overflow: visible;
        width: 100%;
        box-sizing: border-box;
    }

    /* Animated background */
    .login-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255, 77, 79, 0.12) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
        pointer-events: none;
        z-index: 0;
    }

    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    /* Title section */
    .login-title {
        text-align: center;
        margin-bottom: 2.5rem;
        position: relative;
        z-index: 1;
    }

    .login-title h1 {
        color: #ff4d4f;
        font-size: clamp(32px, 6vw, 44px);
        margin-bottom: 0.75rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        text-shadow: 0 2px 20px rgba(255, 77, 79, 0.4);
        animation: glow 2s ease-in-out infinite;
    }

    @keyframes glow {
        0%, 100% { text-shadow: 0 2px 20px rgba(255, 77, 79, 0.4); }
        50% { text-shadow: 0 2px 30px rgba(255, 77, 79, 0.6); }
    }

    .login-subtitle {
        color: #cbd5e1;
        font-size: clamp(14px, 2vw, 16px);
        margin-top: 0.5rem;
        font-weight: 400;
        opacity: 0.95;
        line-height: 1.5;
    }

    /* Slide up animation */
    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(40px) scale(0.96);
        }
        to {
            opacity: 1;
            transform: translateY(0) scale(1);
        }
    }

    /* Responsive breakpoints */
    @media (max-width: 768px) {
        .login-card {
            padding: 2.5rem 2rem;
            border-radius: 20px;
        }

        .login-title {
            margin-bottom: 2rem;
        }

        .login-title h1 {
            font-size: 36px;
        }
    }

    @media (max-width: 480px) {
        .block-container {
            padding: 1.5rem 1rem !important;
        }

        .login-card {
            padding: 2rem 1.5rem;
            border-radius: 18px;
        }

        .login-title {
            margin-bottom: 1.75rem;
        }

        .login-title h1 {
            font-size: 30px;
        }

        .login-subtitle {
            font-size: 14px;
        }
    }

    @media (max-width: 360px) {
        .login-card {
            padding: 1.75rem 1.25rem;
        }

        .login-title h1 {
            font-size: 26px;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    # Login wrapper
    st.markdown('<div class="login-wrapper"><div class="login-card">', unsafe_allow_html=True)

    # Title
    st.markdown("""
    <div class="login-title">
        <h1>🔐 Login</h1>
        <p class="login-subtitle">Access your classroom lectures and AI-powered learning</p>
    </div>
    """, unsafe_allow_html=True)

    # Enhanced input styling
    st.markdown("""
    <style>
    /* Enhanced login input boxes */
    div[data-testid="stTextInput"] {
        position: relative;
        z-index: 1;
    }

    div[data-testid="stTextInput"] input {
        height: 56px !important;
        font-size: 16px !important;
        padding: 16px 20px 16px 48px !important;
        background: rgba(42, 47, 58, 0.9) !important;
        border: 2px solid rgba(255, 77, 79, 0.2) !important;
        border-radius: 12px !important;
        color: #e5e7eb !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        font-weight: 500 !important;
        width: 100% !important;
        box-sizing: border-box !important;
        max-width: 100% !important;
    }

    div[data-testid="stTextInput"] input:hover {
        border-color: rgba(255, 77, 79, 0.4) !important;
        background: rgba(50, 56, 70, 0.9) !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(255, 77, 79, 0.1) !important;
    }

    div[data-testid="stTextInput"] input:focus {
        border-color: #ff4d4f !important;
        background: rgba(50, 56, 70, 1) !important;
        box-shadow: 0 0 0 4px rgba(255, 77, 79, 0.12),
                    0 8px 20px rgba(255, 77, 79, 0.15) !important;
        transform: translateY(-2px);
    }

    div[data-testid="stTextInput"] label {
        font-size: 15px !important;
        font-weight: 600 !important;
        margin-bottom: 10px !important;
        color: #cbd5e1 !important;
        letter-spacing: 0.3px !important;
    }

    /* Enhanced login button */
    div[data-testid="stButton"] {
        margin-top: 1.5rem;
        position: relative;
        z-index: 1;
    }

    div[data-testid="stButton"] button[kind="primary"] {
        height: 56px !important;
        font-size: 17px !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #ff4d4f 0%, #ff6b72 100%) !important;
        border: none !important;
        border-radius: 12px !important;
        color: white !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 16px rgba(255, 77, 79, 0.3) !important;
        position: relative;
        overflow: hidden;
    }

    div[data-testid="stButton"] button[kind="primary"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s;
    }

    div[data-testid="stButton"] button[kind="primary"]:hover::before {
        left: 100%;
    }

    div[data-testid="stButton"] button[kind="primary"]:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 24px rgba(255, 77, 79, 0.4) !important;
        background: linear-gradient(135deg, #ff6b72 0%, #ff4d4f 100%) !important;
    }

    div[data-testid="stButton"] button[kind="primary"]:active {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(255, 77, 79, 0.3) !important;
    }

    /* Demo credentials hint */
    .demo-hint {
        text-align: center;
        margin-top: 2rem;
        padding: 1rem;
        background: rgba(59, 130, 246, 0.08);
        border-radius: 10px;
        border: 1px solid rgba(59, 130, 246, 0.2);
    }

    .demo-hint p {
        color: #94a3b8;
        font-size: 13px;
        margin: 0.25rem 0;
    }

    .demo-hint strong {
        color: #60a5fa;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

    # Form inputs
    username = st.text_input("👤 Username", placeholder="Enter your username", key="login_username")
    password = st.text_input("🔑 Password", type="password", placeholder="Enter your password", key="login_password")

    # Login button
    login_btn = st.button("🚀 Login to Dashboard", use_container_width=True, type="primary")

    # Demo credentials
    st.markdown("""
    <div class="demo-hint">
        <p><strong>Demo Accounts:</strong></p>
        <p>Student: <strong>stu1</strong> / <strong>stu123</strong></p>
        <p>Staff: <strong>staff1</strong> / <strong>staff123</strong></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div></div>', unsafe_allow_html=True)

    # ================== AUTH LOGIC ==================
    if login_btn:
        with open("users.json") as f:
            users = json.load(f)

        if username in users and users[username]["password"] == password:
            st.session_state.logged_in = True
            st.session_state.user = username
            st.session_state.role = users[username]["role"]
            
            # Store session in URL query params for persistence across page navigation
            st.query_params["user"] = username
            st.query_params["role"] = users[username]["role"]
            st.query_params["page"] = "dashboard"
            
            st.success("✅ Login successful! Redirecting...")
            st.rerun()
        else:
            st.error("❌ Invalid credentials. Please try again.")

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

# ================== AI CHAT (HYBRID KNOWLEDGE) ==================

def is_greeting(text):
    greetings = ["hi", "hai", "hello", "hey", "hii"]
    return text.lower().strip() in greetings


if menu == "🤖 AI Chat":
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

    # Chat header
    st.markdown("""
    <div class="chat-header">
        <h1>🤖 Classroom AI Chat</h1>
        <p>Powered by Gemini - Ask questions about your lectures or general topics</p>
    </div>
    """, unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show empty state if no messages
    if len(st.session_state.messages) == 0:
        st.markdown("""
        <div class="chat-empty-state">
            <h3>💬 Start a Conversation</h3>
            <p>Ask me anything about your classroom lectures or general knowledge!</p>
            <p style="margin-top: 1rem; font-size: 14px;">
                Try: "What is AI?" or "Explain machine learning basics"
            </p>
        </div>
        """, unsafe_allow_html=True)

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
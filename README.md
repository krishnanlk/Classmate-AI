# Classmate-AI

An intelligent AI-powered educational assistant that leverages Google Gemini API to provide real-time support for student learning. The application integrates lecture content management, chat persistence, and semantic search capabilities to deliver personalized educational experiences.

## 🎯 Features

- **AI-Powered Chat**: Interactive chat interface powered by Google Gemini API
- **Lecture Management**: Organize and manage course lectures by subject and unit
- **Document Upload**: Upload PDF and Word documents to ask questions about their content
- **Smart Q&A**: Ask questions about lectures, uploaded documents, or general knowledge
- **Semantic Search**: TF-IDF based content similarity matching across lecture materials
- **Multi-Subject Support**: Handle multiple courses (AI, DAA, DBMS, etc.)
- **Cloud Storage Integration**: Structured cloud storage for organized content management
- **Downloadable Notes**: Generate lecture notes as PDF or Word documents
- **Multi-User Support**: Separate profiles for students and staff with role-based features
- **Responsive UI**: Clean, modern interface built with Streamlit for seamless user experience

## 📋 Prerequisites

Before deploying Classmate-AI, ensure you have the following installed:

- **Python**: Version 3.8 or higher
- **pip**: Python package manager
- **FFmpeg**: Required for audio processing
- **Git**: For version control
- **Google Gemini API Key**: Obtain from [Google AI Studio](https://makersuite.google.com/app/apikey)

### System Requirements

- **OS**: Windows, macOS, or Linux
- **RAM**: Minimum 4GB (8GB recommended)
- **Disk Space**: At least 2GB for dependencies and cache
- **Internet**: Required for API calls and updates

## 🚀 Deployment Guide

### Step 1: Clone the Repository

```bash
git clone https://github.com/krishnanlk/Classmate-AI.git
cd Classmate-AI
```

### Step 2: Create Virtual Environment

Creating a virtual environment isolates project dependencies:

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Dependency Overview:**
- `streamlit` (1.51.0+): Web framework for the UI
- `google-generativeai` (0.8.6+): Google Gemini API integration
- `openai-whisper`: Speech-to-text processing
- `torch`: Deep learning framework for ML models
- `transformers`: Pre-trained model library
- `pytube`: YouTube video downloading
- `ffmpeg-python`: Audio/video processing
- `scikit-learn`: Machine learning utilities for similarity matching and TF-IDF vectorization
- `reportlab` (4.4.9+): PDF document generation with professional styling
- `python-docx` (1.2.0+): Word document (.docx) creation and manipulation
- `PyPDF2` (3.0.1+): PDF text extraction and processing
- `numpy`: Numerical computing library
- `wikipedia`: External knowledge base for general questions

### Step 4: Configure Environment Variables

Create a `.env` file in the project root directory with your Google Gemini API key:

```bash
cp gemini.env .env  # Create a copy if template exists
```

Edit the `.env` file:

```env
GEMINI_API_KEY=your_actual_api_key_here
```

**To obtain your Gemini API key:**
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click "Create API Key"
3. Copy the generated key
4. Paste it in the `.env` file

⚠️ **Security Note**: Never commit `.env` files with real API keys to version control. The `.env` file should be in `.gitignore`.

### Step 5: Verify Installation

Test if all dependencies are correctly installed:

```bash
python -m pytest test_gemini.py  # If pytest is available
# OR manually test
python -c "import streamlit; import torch; import transformers; print('All dependencies installed successfully!')"
```

### Step 6: Run the Application

Start the Streamlit web server:

```bash
streamlit run app.py
```

The application will be accessible at: **http://localhost:8501**

## 📁 Project Structure

```
Classmate-AI/
├── app.py                      # Main Streamlit application
├── gemini_chat.py              # Gemini API integration & chat logic
├── gemini_config.py            # Configuration for Gemini settings
├── connect.py                  # Database/lecture connection utilities
├── document_extractor.py       # PDF/Word document text extraction
├── notes_generator.py          # PDF/Word lecture notes generation
├── test_gemini.py              # Unit tests for Gemini functionality
├── requirements.txt            # Python dependencies
├── gemini.env                  # Environment variable template
├── users.json                  # User data storage
│
├── cloud_storage/              # Organized content storage
│   ├── AI/
│   │   ├── Unit_1/
│   │   │   └── 2026-01-07/
│   │   │       └── AI__Unit_1_YouTube_tutorial_video_00-59.txt
│   │   └── ...
│   ├── DAA/
│   │   ├── unit_1/
│   │   │   └── 2026-01-16/
│   │   │       └── DAA_unit_1_Time_Complexities_in_Algorithms__00-41.txt
│   │   └── ...
│   └── DBMS/
│       ├── unit_1/
│       └── ...
│
├── lectures/                   # Additional lecture materials
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## ⚙️ Configuration Details

### app.py Configuration

The main application handles:
- **Page Layout**: Wide layout optimized for desktop viewing
- **Storage Paths**: `cloud_storage/` for lecture materials
- **User Management**: Separate student and staff roles with different features
- **File Upload**: Support for PDF and Word document uploads
- **Document Processing**: Automatic text extraction from uploaded files
- **Lecture-Aware Q&A**: Context-aware answers using lecture content
- **Notes Generation**: Create downloadable PDF and Word documents from lectures

### gemini_config.py

Configure Gemini API settings:
```python
# Default configuration
GEMINI_MODEL = "gemini-pro"
MAX_TOKENS = 2048
TEMPERATURE = 0.7
TOP_P = 0.9
```

### document_extractor.py

Handles document processing:
- **PDF Extraction**: Uses PyPDF2 to extract text from PDF files
- **Word Extraction**: Uses python-docx to extract content from .docx files
- **Page Tracking**: Maintains information about which page content came from

### notes_generator.py

Generates downloadable study materials:
- **PDF Notes**: Professional formatted PDF documents with key concepts
- **Word Notes**: Editable Word documents for student note-taking
- **Key Point Extraction**: Uses Gemini AI to identify important concepts from lectures

## 📊 Features in Detail

### Chat Interface (Students Only)

The chat interface provides:
- **Simple, Clean Design**: Minimalist interface for focused learning
- **Document Upload**: Attach PDF or Word files using the "Attach File" button
- **Context-Aware Responses**: Answers based on lectures, uploaded documents, or general knowledge
- **Source Attribution**: Clearly shows where answers come from (📘 Classroom Lectures, 📄 Uploaded Document, 🌐 General Knowledge)

### Lecture Viewer

Browse and study your course materials:
- **Subject-Based Organization**: Find lectures by course (AI, DAA, DBMS, etc.)
- **Unit Navigation**: Access specific units and topics
- **Searchable Content**: Use semantic search to find relevant lecture materials
- **Download Notes**: Generate and download lecture notes as PDF or Word documents

### Role-Based Features

**Students Can:**
- ✅ View lectures by subject and unit
- ✅ Chat with AI about lecture content
- ✅ Upload documents to ask questions
- ✅ Download lecture notes as PDF/Word

**Staff Can:**
- ✅ Upload new lectures to the system
- ✅ Manage course content
- ✅ All student features

### Storage Structure

Lectures are organized in `cloud_storage/`:
```
cloud_storage/
├── [SUBJECT]/
│   ├── [UNIT]/
│   │   └── [DATE]/
│   │       └── [lecture_file].txt
```

Example:
```
cloud_storage/AI/Unit_1/2026-01-07/AI__Unit_1_YouTube_tutorial_video_00-59.txt
cloud_storage/DAA/unit_1/2026-01-16/DAA_unit_1_Time_Complexities_in_Algorithms__00-41.txt
```

## 🔧 Advanced Configuration

### Port Configuration

To run on a custom port:

```bash
streamlit run app.py --server.port 8080
```

### Deployment on Server

**Using Gunicorn (Production):**
```bash
pip install gunicorn
gunicorn app:app --bind 0.0.0.0:8000
```

**Using Docker (Optional):**

Create a `Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["streamlit", "run", "app.py"]
```

Build and run:
```bash
docker build -t classmate-ai .
docker run -p 8501:8501 -e GEMINI_API_KEY=your_key classmate-ai
```

## 🐛 Troubleshooting

### Issue: Module not found errors
**Solution**: Ensure virtual environment is activated and all dependencies are installed
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Issue: API Key errors
**Solution**: Verify `.env` file exists and contains valid Gemini API key
```bash
cat .env  # Verify content
```

### Issue: FFmpeg not found
**Solution**: Install FFmpeg based on your OS
- **Windows**: `choco install ffmpeg` or download from [ffmpeg.org](https://ffmpeg.org)
- **macOS**: `brew install ffmpeg`
- **Linux**: `sudo apt-get install ffmpeg`

### Issue: Port already in use
**Solution**: Specify a different port
```bash
streamlit run app.py --server.port 8080
```

## 📝 Usage Guide

### For Students

#### 1. Starting the Application
```bash
streamlit run app.py
```
Then access at `http://localhost:8501`

#### 2. Login/Authentication
- Enter your user ID in the sidebar
- Select "Student" role
- Click "Enter Chat" or "View Lectures"

#### 3. Reading Lectures
1. Click **📺 View Lectures** in the top menu
2. Select a subject from the dropdown
3. Choose a unit to explore
4. View lecture content and text
5. **Download Notes**: Click "📥 Download as PDF" or "📄 Download as Word" for study materials

#### 4. Asking Questions via Chat
1. Click **🤖 AI Chat** in the top menu
2. Type your question about:
   - Lecture content (if lecture is loaded)
   - Uploaded documents
   - General knowledge topics
3. Optional: Click **Attach File** to upload a PDF or Word document
4. Press Enter to get AI-generated answers with source attribution

#### 5. Document-Based Q&A
1. In the chat interface, click **Attach File**
2. Upload a PDF or Word document
3. Ask questions about the document content
4. AI will extract and search the document for relevant answers

### Example Workflows

**Workflow 1: Study with Lecture Notes**
```
1. Open Lecture Viewer
2. Select AI > Unit 1 > 2026-01-07
3. Read AI__Unit_1_YouTube_tutorial_video_00-59.txt
4. Click "Download as PDF" to get study notes
```

**Workflow 2: Ask Questions About Lecture**
```
1. Open AI Chat
2. Ask: "What are the key concepts in machine learning?"
3. AI searches lectures and provides context-aware answer
4. See source: "📘 Classroom Lectures"
```

**Workflow 3: Study External Material**
```
1. Open AI Chat
2. Click "Attach File"
3. Upload your research paper (PDF)
4. Ask: "Summarize the main findings"
5. AI extracts from document and summarizes
6. See source: "📄 [document_name]"
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Krishnan LK**
- GitHub: [@krishnanlk](https://github.com/krishnanlk)

## 📞 Support & Issues

For issues, questions, or suggestions:
1. Check existing [GitHub Issues](https://github.com/krishnanlk/Classmate-AI/issues)
2. Create a new issue with detailed description
3. Provide error messages and reproduction steps

## 🔒 Security Notes

- **Never commit `.env` files** with real API keys to version control
- Keep `gemini.env` as template only - never upload with actual keys
- Use environment variables for all sensitive data in production
- Regularly rotate API keys to prevent unauthorized access
- Monitor API usage to detect anomalies
- Excluded files (in .gitignore):
  - Environment files: `.env`, `*.env`, `gemini.env`
  - Python cache: `__pycache__/`, `*.pyc`, `*.pyo`
  - Virtual environments: `venv/`, `env/`
  - IDE configs: `.vscode/`, `.idea/`
  - Build artifacts: `dist/`, `build/`, `*.egg-info`

## 🎓 Educational Purpose

Classmate-AI is designed to enhance student learning through AI assistance. Features include:
- Independent learning support
- Study material generation
- Homework assistance
- Concept clarification
- Document analysis

**Note**: This tool is intended for students only. Features are restricted by role-based access control.

## 📈 Roadmap

<<<<<<< HEAD
- [ ] **Lecture-Aware Quiz Generator** - Auto-generate MCQs, Short Answer, and True/False questions from lectures (Students only)
- [ ] Advanced analytics dashboard for learning progress tracking
- [ ] Multi-language support for international students
- [ ] Voice input/output capabilities
- [ ] Mobile app version
- [ ] Offline mode support for lectures
- [ ] Custom model fine-tuning for specific domains
- [ ] Collaborative study features for group learning
- [ ] Real-time assignment feedback and grading
- [ ] Integration with learning management systems (LMS)

## 🔄 Recent Updates (January 2026)

### Version 1.1.0 - Simplified Student Interface

**New Features:**
- ✨ **Document Upload**: Upload PDF and Word documents to ask questions about their content
- 📝 **Lecture Notes Generator**: Download lectures as PDF or Word documents
- 📎 **File Attachment Button**: Clean, simple file upload interface in chat
- 🎯 **Smart Context Integration**: AI answers combine lecture content, uploaded documents, and general knowledge
- 📊 **Source Attribution**: Clear indication of where each answer came from

**UI/UX Improvements:**
- 🎨 Simplified chat interface focused on student learning
- ❌ Removed chat history sidebar for cleaner appearance
- 📌 Made attach file button prominently visible
- 🚀 Faster, more responsive interactions

**Code Quality:**
- 🔍 Improved error handling for document processing
- 📦 Enhanced document extraction for PDF and Word formats
- 🎯 Better semantic search across lecture materials
- 🔐 Updated .gitignore with comprehensive exclusion rules

---

**Last Updated**: January 24, 2026  
**Current Version**: 1.1.0  
**Author**: Krishnan LK  
**Repository**: [krishnanlk/Classmate-AI](https://github.com/krishnanlk/Classmate-AI)  
**Status**: ✅ Active Development

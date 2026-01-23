# Classmate-AI

An intelligent AI-powered educational assistant that leverages Google Gemini API to provide real-time support for student learning. The application integrates lecture content management, chat persistence, and semantic search capabilities to deliver personalized educational experiences.

## 🎯 Features

- **AI-Powered Chat**: Interactive chat interface powered by Google Gemini API
- **Lecture Management**: Organize and manage course lectures by subject and unit
- **Chat History**: Persistent chat history per user with JSON storage
- **Semantic Search**: TF-IDF based content similarity matching
- **Multi-Subject Support**: Handle multiple courses (AI, DAA, DBMS, etc.)
- **Cloud Storage Integration**: Structured cloud storage for organized content management
- **Responsive UI**: Built with Streamlit for seamless user experience

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
- `streamlit`: Web framework for the UI
- `openai-whisper`: Speech-to-text processing
- `torch`: Deep learning framework for ML models
- `transformers`: Pre-trained model library
- `pytube`: YouTube video downloading
- `ffmpeg-python`: Audio/video processing
- `scikit-learn`: Machine learning utilities for similarity matching
- `numpy`: Numerical computing library

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
├── app.py                    # Main Streamlit application
├── gemini_chat.py           # Gemini API integration & chat logic
├── gemini_config.py         # Configuration for Gemini settings
├── connect.py               # Database/lecture connection utilities
├── test_gemini.py           # Unit tests for Gemini functionality
├── requirements.txt         # Python dependencies
├── gemini.env              # Environment variable template
├── users.json              # User data storage
│
├── chat_history/           # User chat persistence
│   ├── default_user.json
│   └── staff1.json
│
├── cloud_storage/          # Organized content storage
│   ├── AI/
│   │   └── UNIT_1/
│   │       └── 2026-01-14/
│   │           └── AI_UNIT_1_Sample_topic_01-25PM.txt
│   ├── DAA/
│   │   └── Unit-1/
│   │       └── 2026-01-14/
│   │           └── DAA_Unit-1_Alogorithm__12-56PM.txt
│   ├── DBMS/
│   │   └── UNIT_2/
│   └── Test/
│
├── lectures/               # Additional lecture materials
└── README.md              # This file
```

## ⚙️ Configuration Details

### app.py Configuration

The main application handles:
- **Page Layout**: Set to wide layout for better UI
- **Storage Paths**: `cloud_storage/` and `chat_history/` directories
- **Chat Persistence**: Automatic save/load of user conversations
- **User Management**: User IDs for personalized experiences

### gemini_config.py

Configure Gemini API settings:
```python
# Example configuration structure
GEMINI_MODEL = "gemini-pro"
MAX_TOKENS = 2048
TEMPERATURE = 0.7
TOP_P = 0.9
```

### Environment Setup

Key environment variables:
```env
GEMINI_API_KEY=your_api_key           # Required for API calls
STREAMLIT_SERVER_PORT=8501            # Optional: custom port
STREAMLIT_LOGGER_LEVEL=info           # Optional: logging level
```

## 📊 Database/Storage

### Chat History Format

Chat histories are stored as JSON files in `chat_history/`:

```json
[
  {
    "role": "user",
    "content": "What is machine learning?",
    "timestamp": "2026-01-20T10:30:00"
  },
  {
    "role": "assistant",
    "content": "Machine learning is...",
    "timestamp": "2026-01-20T10:30:05"
  }
]
```

### Cloud Storage Organization

Content is organized by:
- **Subject**: AI, DAA, DBMS, Test, etc.
- **Unit/Chapter**: UNIT_1, Unit-1, UNIT_2, etc.
- **Date**: Organized by date of creation (YYYY-MM-DD)

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

## 📝 Usage Examples

### Starting a Chat Session

1. Launch the application: `streamlit run app.py`
2. Enter your user ID in the sidebar
3. Begin typing your questions about courses

### Managing Lectures

Lectures should be stored in the `cloud_storage/` directory with the structure:
```
cloud_storage/[SUBJECT]/[UNIT]/[DATE]/[content_file]
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

- **Never commit `.env` files** with real API keys
- Keep `gemini.env` updated with API key template only
- Use environment variables for sensitive data in production
- Regularly rotate API keys
- Monitor API usage to detect unauthorized access

## 🎓 Educational Purpose

Classmate-AI is designed to enhance student learning through AI assistance. Use responsibly and supplement traditional learning methods.

## 📈 Roadmap

- [ ] Voice input/output capabilities
- [ ] Advanced analytics dashboard

---

**Last Updated**: January 2026 | **Version**: 1.0.0

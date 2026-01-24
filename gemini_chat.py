from gemini_config import client

def gemini_chat(question, lecture_context=None):
    """
    Works with:
    gemini_chat(prompt)
    gemini_chat(prompt, lecture_context)
    """

    # If lecture context is provided and meaningful, build strict prompt
    if lecture_context and len(lecture_context.strip()) > 150:
       prompt = f"""
You are Classroom AI.

STRICT RULES:
- Answer ONLY using the lecture content provided below.
- DO NOT use general knowledge.
- DO NOT imagine or assume any lecture.
- If the lecture content is empty or insufficient, reply EXACTLY with:
  "NO_LECTURE_FOUND"

Lecture Content:
{lecture_context}

Question:
{question}

Answer format:
📘 Title
• Point 1
• Point 2
• Point 3
"""

    else:
        # Either external knowledge OR app.py already built the prompt
        prompt = question

    # Call Gemini with error handling
    try:
        response = client.generate_content(prompt)
        
        # Check if response has valid content
        if response.parts and len(response.parts) > 0:
            return response.text.strip()
        else:
            # Handle empty or blocked response
            return "I couldn't generate a response. This might be due to content filtering or API limitations. Please try again with a different question."
    
    except ValueError as e:
        # Handle API errors (blocked content, rate limits, etc.)
        error_msg = str(e)
        if "finish_reason" in error_msg or "response" in error_msg.lower():
            return "The API returned an empty response. This could mean:\n• Your question was filtered by safety systems\n• API rate limit exceeded\n• Please try rephrasing your question or try again later."
        else:
            return f"API Error: {error_msg}"
    
    except Exception as e:
        # Catch other unexpected errors
        return f"An error occurred: {str(e)}"

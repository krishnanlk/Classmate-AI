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

    # Call Gemini
    response = client.generate_content(prompt)

    return response.text.strip()

import os

def load_all_lectures():
    lecture_text = ""
    base_path = "lectures"

    if not os.path.exists(base_path):
        return "No lectures uploaded yet."

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".txt"):
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    lecture_text += f"\n\n--- {file} ---\n"
                    lecture_text += f.read()

    if lecture_text.strip() == "":
        return "Lecture files are empty."

    return lecture_text

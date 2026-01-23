from PyPDF2 import PdfReader
from docx import Document
import os

def extract_text_from_pdf(pdf_file):
    """
    Extract text from a PDF file.
    
    Args:
        pdf_file: File object or path to PDF file
    
    Returns:
        str: Extracted text from PDF
    """
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += f"\n--- Page {page_num + 1} ---\n"
            text += page.extract_text()
        return text.strip()
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")


def extract_text_from_word(docx_file):
    """
    Extract text from a Word document.
    
    Args:
        docx_file: File object or path to DOCX file
    
    Returns:
        str: Extracted text from Word document
    """
    try:
        doc = Document(docx_file)
        text = ""
        for para in doc.paragraphs:
            if para.text.strip():
                text += para.text + "\n"
        return text.strip()
    except Exception as e:
        raise Exception(f"Error extracting text from Word document: {str(e)}")


def extract_text_from_document(file, file_type):
    """
    Extract text from either PDF or Word document based on file type.
    
    Args:
        file: Uploaded file object
        file_type: Type of file ('pdf' or 'docx')
    
    Returns:
        tuple: (extracted_text, filename)
    """
    filename = file.name
    
    if file_type.lower() == 'pdf':
        extracted_text = extract_text_from_pdf(file)
    elif file_type.lower() in ['docx', 'doc']:
        extracted_text = extract_text_from_word(file)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    return extracted_text, filename

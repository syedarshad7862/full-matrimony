import pdfplumber
import pdb

def extract_text_from_pdf(pdf_file) -> str:
    """Extracts text from a PDF file using pdfplumber.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text from the PDF.
    """
    text = ""
    
    try:
        with pdfplumber.open(pdf_file.file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n" if page.extract_text() else ""
        # print(pdf_file)
    
        # pdb.set_trace()
                
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""
    
    return text

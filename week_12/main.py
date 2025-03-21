import os
import pdfplumber
import docx
import openai

# Set your OpenAI API key
OPENAI_API_KEY = "My GPT Key"
openai.api_key = OPENAI_API_KEY

def extract_text_from_pdf(file_path):
    """Extracts text from a PDF file."""
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.strip()

def extract_text_from_docx(file_path):
    """Extracts text from a DOCX file."""
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs]).strip()

def analyze_cv(cv_text):
    """Sends CV text to OpenAI GPT-4.0 for analysis."""
    prompt = f"""
    You are an expert AI recruiter analyzing a candidate's CV in IT, software engineering, data analytics and computer science fields. 

    1. Identify and categorize the candidate's experience into fields (e.g., Software Engineering, Lecturer, Business, Finance).
    2. Suggest the main two areas of the candidate's expertise and the most relevant job roles based on the experience.
    3. Provide recommendations for improving the CV in three bullet points.

    CV Text:
    {cv_text}
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a professional recruiter analyzing resumes."},
                  {"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"]

if __name__ == "__main__":
    file_path = input("Enter CV file path (PDF/DOCX): ").strip()

    if not os.path.exists(file_path):
        print("File not found!")
        exit()

    # Extract text based on file type
    if file_path.endswith(".pdf"):
        cv_text = extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        cv_text = extract_text_from_docx(file_path)
    else:
        print("Unsupported file format!")
        exit()

    print("\nAnalyzing CV with GPT-4.0...\n\n ", cv_text,"\n\n")
    analysis_result = analyze_cv(cv_text)
    
    print("\n--- CV Analysis Results ---\n")
    print(analysis_result)

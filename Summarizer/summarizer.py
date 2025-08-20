from transformers import pipeline
import pdfplumber 
summarizer = pipeline("summarization", model="google-t5/t5-small")

def read_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() 
    return text


file_path = r'C:\Users\ADMIN\Documents\GitHub\Transformers\Summarizer\Docs\Photosynthesis.pdf'   
pdf = read_pdf(file_path)

summary = summarizer(pdf, max_length=130, min_length=50, do_sample=False)

print(summary)
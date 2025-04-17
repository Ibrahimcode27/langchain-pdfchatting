import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
import google.generativeai as genai
from pdf2image import convert_from_path
from PIL import Image
from paddleocr import PaddleOCR
import fitz
from docx import Document

# ✅ Set your Gemini API Key
os.environ["GEMINI_API_KEY"] = "AIzaSyCmYCpSv-0G1oaEuTcO8cLy91JfmB3AjA0"
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ✅ Initialize PaddleOCR Model
ocr = PaddleOCR(use_angle_cls=True, lang="en")

# ✅ Function to Convert PDF Pages to Images
def convert_pdf_to_images(pdf_path):
    images = convert_from_path(pdf_path)
    image_paths = []
    for i, img in enumerate(images):
        img_path = f"page_{i + 1}.jpg"
        img.save(img_path, 'JPEG')
        image_paths.append(img_path)
    return image_paths

# ✅ Function to Extract Text from Images using PaddleOCR
def extract_text_from_image(image_path):
    result = ocr.ocr(image_path, cls=True)
    extracted_text = "\n".join([line[1][0] for line in result[0] if line[1]])
    return extracted_text

# ✅ Function to Extract Text from PDFs using PyMuPDF (Direct Method)
def extract_text_from_pdf_direct(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text() + "\n"
    return text.strip()

# ✅ Function to Extract Text from PDFs using PaddleOCR (OCR Method)
def extract_text_from_pdf_ocr(pdf_path):
    image_paths = convert_pdf_to_images(pdf_path)
    extracted_texts = [extract_text_from_image(img) for img in image_paths]
    return "\n".join(extracted_texts)

# ✅ Function to Extract Text from DOCX
def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text.strip()

# ✅ Function to Process Uploaded Files (PDF → OCR or Direct, DOCX → Direct)
def process_uploaded_file(file_path):
    if file_path.endswith(".pdf"):
        extracted_text = extract_text_from_pdf_direct(file_path)
        if len(extracted_text.strip()) < 10:
            print("⚠️ Direct text extraction failed, switching to OCR...")
            extracted_text = extract_text_from_pdf_ocr(file_path)
    elif file_path.endswith(".docx"):
        extracted_text = extract_text_from_docx(file_path)
    else:
        raise ValueError("Unsupported file format. Please upload a PDF or DOCX file.")

    with open("extracted_text.txt", "w", encoding="utf-8") as f:
        f.write(extracted_text)

    return extracted_text

# ✅ Load and Process Document
file_path = "Magnetic Effects of Current and Magnetism JEE Main Questions FREE PDF.pdf"
text = process_uploaded_file(file_path)

# ✅ Tokenizer Setup
def count_tokens(text: str) -> int:
    return len(text.split())

# ✅ Splitting the Text into Chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=24,
    length_function=count_tokens,
)

chunks = text_splitter.create_documents([text])
token_counts = [count_tokens(chunk.page_content) for chunk in chunks]
df = pd.DataFrame({'Token Count': token_counts})
df.hist(bins=40)
plt.show()

# ✅ Using Sentence Transformers for Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(chunks, embeddings)

# ✅ ✅ **Fix: Wrapping Gemini API inside an LLM class**
class GeminiLLM(LLM):
    """A wrapper to use Gemini API as an LLM in LangChain."""
    
    def _call(self, prompt, stop=None):
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        return response.text.strip()

    @property
    def _llm_type(self):
        return "gemini"

# ✅ Initialize the Gemini LLM Wrapper
gemini_llm = GeminiLLM()

# ✅ Conversational Chain
qa = RetrievalQA.from_chain_type(llm=gemini_llm, retriever=db.as_retriever(), chain_type="stuff")

# ✅ Chatbot Interaction Loop
chat_history = []
def chatbot():
    print("\nWelcome to the Document Chatbot! Type 'exit' to stop.\n")

    while True:
        query = input("User: ")

        if query.lower() == 'exit':
            print("\nThank you for using the chatbot!")
            break

        answer = qa.run(query)

        chat_history.append((query, answer))
        print(f"\nChatbot: {answer}\n")

# ✅ Start Chatbot
chatbot()

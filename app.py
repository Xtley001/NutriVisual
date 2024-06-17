### NutriVisual App - Diet Composition Analyzer
from dotenv import load_dotenv
import streamlit as st
import os
import google.generativeai as genai
from PIL import Image
import fitz  # PyMuPDF for PDF handling
import textract  # textract for DOC handling
import matplotlib.pyplot as plt
import pandas as pd

load_dotenv()  ## load all the environment variables

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

## Function to load Google Gemini Pro Vision API And get response
def get_gemini_response(image):
    input_prompt = """
    Diet Composition Analyzer
    
    Analyze the diet composition based on the provided document or image. 
    Please identify the types of food and their composition.
    
    Format for Reporting:
    
    - List each food item followed by its composition details (e.g., macros, vitamins).
    - Provide insights into the overall diet balance and nutritional value.
    
    Example:
    
    Protein-rich foods - 40%
    Carbohydrates - 30%
    Vegetables - 20%
    Fruits - 10%
    """
    model = genai.GenerativeModel('gemini-pro-vision')
    response = model.generate_content(["", image[0], input_prompt])
    return response.text

def input_file_setup(uploaded_file):
    # Check if a file has been uploaded
    if uploaded_file is not None:
        # Handle different file types
        file_extension = uploaded_file.name.split(".")[-1].lower()
        if file_extension in ['pdf']:
            # Process PDF using PyMuPDF (fitz)
            pdf_doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            images = []
            for page_num in range(len(pdf_doc)):
                page = pdf_doc.load_page(page_num)
                images.append(Image.frombytes("RGB", [page.width, page.height], page.get_pixmap().tobytes()))
            return images
        elif file_extension in ['doc', 'docx']:
            # Process DOC/DOCX using textract
            extracted_text = textract.process(uploaded_file)
            # For simplicity, assuming the first page/image is processed
            image = Image.new('RGB', (1, 1))  # Placeholder image
            return [image]
        elif file_extension in ['jpg', 'jpeg', 'png']:
            # Process image file
            image = Image.open(uploaded_file)
            return [image]
        else:
            raise ValueError(f"Unsupported file format: {file_extension}. Please upload a PDF, DOC, DOCX, JPG, JPEG, or PNG file.")
    else:
        raise FileNotFoundError("No file uploaded")

## Initialize our Streamlit app
st.set_page_config(page_title="NutriVisual")

st.title("NutriVisual")
st.markdown("""
Welcome to NutriVisual! This tool helps nutritionists and dietitians analyze diet composition based on uploaded documents or images.
""")

uploaded_file = st.file_uploader("Choose a file (PDF, DOC, JPG, JPEG, PNG)...", type=["pdf", "doc", "docx", "jpg", "jpeg", "png"])
image = None
if uploaded_file is not None:
    try:
        image = input_file_setup(uploaded_file)
        st.image(image[0], caption="Uploaded File Preview.", use_column_width=True)
    except Exception as e:
        st.error(f"Error: {e}")

submit = st.button("Tell me about this diet and it's composition")

## If submit button is clicked
if submit and image:
    response = get_gemini_response(image)
    st.subheader("Analysis Results")
    st.write(response)

    # Assuming the response is in the correct format and parsing it
    def parse_response(response):
        lines = response.split('\n')
        items = []
        for line in lines:
            if '-' in line:
                item, details = line.split('-')
                items.append((item.strip(), details.strip()))
        return items

    items = parse_response(response)

    # Displaying the parsed results
    st.subheader("Diet Composition Analysis")
    for item in items:
        st.write(f"- {item[0]}: {item[1]}")


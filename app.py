from dotenv import load_dotenv
import streamlit as st
import os
import google.generativeai as genai
from PIL import Image
import fitz  # PyMuPDF for PDF handling
import textract  # textract for DOC handling
import matplotlib.pyplot as plt

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
                # Get dimensions of the page
                rect = page.rect
                width = int(rect.width)
                height = int(rect.height)
                images.append(Image.frombytes("RGB", (width, height), page.get_pixmap().tobytes()))
            return images
        elif file_extension in ['doc', 'docx']:
            # Process DOC/DOCX using textract
            extracted_text = textract.process(uploaded_file)
            # For simplicity, returning a placeholder image
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

## Function to analyze diet composition and generate pie chart
def analyze_diet_and_generate_chart(response):
    # Parse response
    lines = response.split('\n')
    food_classes = {}

    for line in lines:
        line = line.strip()
        if line and '-' in line:
            parts = line.split('-')
            if len(parts) == 2:
                item = parts[0].strip()
                details = parts[1].strip()
                if details.endswith('%'):
                    try:
                        percentage = float(details[:-1])  # Remove '%' and convert to float
                        food_classes[item] = percentage
                    except ValueError:
                        continue  # Skip if percentage cannot be converted to float

    if not food_classes:
        st.warning("No valid food composition data found in the analysis results.")
        return {}

    # Generate pie chart
    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(food_classes.values(), labels=food_classes.keys(), autopct='%1.1f%%', startangle=140)

    # Add percentage labels inside pie chart
    for text in texts:
        text.set_fontsize(12)
    for autotext in autotexts:
        autotext.set_fontsize(12)

    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)

    return food_classes

## Function to provide tailored recommendations based on condition
def get_tailored_recommendations(patient_condition):
    recommendations = {
        'General': """
            Recommendations for General Health:
            - Maintain a balanced diet with adequate portions of protein, carbohydrates, and vegetables.
            - Monitor overall calorie intake based on activity level.
            """,
        'Diabetes': """
            Recommendations for Diabetes:
            - Emphasize low glycemic index carbohydrates.
            - Include lean protein sources and healthy fats.
            - Monitor carbohydrate intake to manage blood glucose levels.
            """,
        'Hypertension': """
            Recommendations for Hypertension:
            - Limit sodium intake and choose low-sodium foods.
            - Increase potassium-rich foods such as fruits and vegetables.
            - Focus on whole grains and lean protein sources.
            """,
        'High Cholesterol': """
            Recommendations for High Cholesterol:
            - Choose healthy fats such as avocados, nuts, and olive oil.
            - Include soluble fiber-rich foods like oats, beans, and fruits.
            - Limit saturated fats and cholesterol intake from animal products.
            """,
        'Gluten Intolerance or Celiac Disease': """
            Recommendations for Gluten Intolerance or Celiac Disease:
            - Avoid gluten-containing grains such as wheat, barley, and rye.
            - Opt for gluten-free alternatives like quinoa, rice, and corn.
            - Ensure food choices are labeled gluten-free if processed.
            """,
        'Vegetarian or Vegan Diet': """
            Recommendations for Vegetarian or Vegan Diet:
            - Ensure adequate protein intake from plant-based sources like beans, tofu, and quinoa.
            - Include a variety of fruits, vegetables, and whole grains for essential nutrients.
            - Consider supplementation for vitamins like B12 and D that may be lacking in a vegan diet.
            """,
        'Food Allergies': """
            Recommendations for Food Allergies:
            - Avoid allergenic foods identified through testing.
            - Read food labels carefully to avoid cross-contamination.
            - Substitute allergenic foods with safe alternatives to maintain balanced nutrition.
            """,
        'Low Carb/Ketogenic Diet': """
            Recommendations for Low Carb/Ketogenic Diet:
            - Limit carbohydrate intake to induce ketosis.
            - Focus on healthy fats, moderate protein, and low-carb vegetables.
            - Monitor electrolyte balance and hydration levels.
            """,
        'Athletes or Active Individuals': """
            Recommendations for Athletes or Active Individuals:
            - Adjust calorie intake to support energy expenditure.
            - Include adequate protein for muscle repair and carbohydrates for energy.
            - Stay hydrated and replenish electrolytes during and after exercise.
            """,
        'Pregnancy or Lactation': """
            Recommendations for Pregnancy or Lactation:
            - Ensure sufficient intake of nutrients like folic acid, iron, and calcium.
            - Include protein-rich foods, healthy fats, and a variety of fruits and vegetables.
            - Stay hydrated and monitor weight gain as advised by healthcare providers.
            """,
        'Heart Disease': """
            Recommendations for Heart Disease:
            - Emphasize a heart-healthy diet rich in fruits, vegetables, and whole grains.
            - Choose lean protein sources and limit saturated and trans fats.
            - Monitor sodium intake and choose low-sodium options.
            """,
        'Gastrointestinal Disorders (e.g., IBS)': """
            Recommendations for Gastrointestinal Disorders:
            - Identify trigger foods and avoid known irritants.
            - Include soluble fiber from oats, bananas, and peeled fruits.
            - Consider probiotics and small, frequent meals to manage symptoms.
            """,
        'Renal Disease': """
            Recommendations for Renal Disease:
            - Limit phosphorus and potassium intake from foods like dairy and bananas.
            - Monitor protein intake based on kidney function.
            - Control sodium intake to manage fluid retention and blood pressure.
            """
    }

    return recommendations.get(patient_condition, "No specific recommendations available for this condition.")

## Initialize our Streamlit app
st.set_page_config(page_title="NutriVisual")

st.title("NutriVisual")
st.markdown("""
Welcome to NutriVisual! This tool helps nutritionists and dietitians analyze diet composition based on uploaded documents or images.
""")

# Patient condition selection
patient_conditions = [
    'General',
    'Diabetes',
    'Hypertension',
    'High Cholesterol',
    'Gluten Intolerance or Celiac Disease',
    'Vegetarian or Vegan Diet',
    'Food Allergies',
    'Low Carb/Ketogenic Diet',
    'Athletes or Active Individuals',
    'Pregnancy or Lactation',
    'Heart Disease',
    'Gastrointestinal Disorders (e.g., IBS)',
    'Renal Disease'
]

patient_condition = st.selectbox('Select Patient Condition', patient_conditions)

# Updated input prompt based on selected condition
input_prompts = {
    'General': """
    Diet Composition Analyzer
    
    Analyze the diet composition based on the provided document or image. 
    Please identify the types of food and their composition.
    
    Example:
    
    Protein-rich foods - 40
    Protein-rich foods - 40%
    Carbohydrates - 30%
    Vegetables - 20%
    Fruits - 10%
    """,
    'Diabetes': """
    Diet Composition Analyzer for Diabetes Patients
    
    Analyze the diet composition based on the provided document or image, considering diabetes management.
    
    Example:
    
    Low glycemic index carbohydrates - 50%
    Lean protein sources - 30%
    Healthy fats - 20%
    """,
    'Hypertension': """
    Diet Composition Analyzer for Hypertension Patients
    
    Analyze the diet composition based on the provided document or image, focusing on hypertension management.
    
    Example:
    
    Low sodium foods - 50%
    Potassium
    -rich foods - 30%
    Whole grains - 20%
    """,
    'High Cholesterol': """
    Diet Composition Analyzer for High Cholesterol Patients
    
    Analyze the diet composition based on the provided document or image, focusing on managing high cholesterol levels.
    
    Example:
    
    Healthy fats - 40%
    Soluble fiber-rich foods - 30%
    Lean protein sources - 30%
    """,
    'Gluten Intolerance or Celiac Disease': """
    Diet Composition Analyzer for Gluten Intolerance or Celiac Disease
    
    Analyze the diet composition based on the provided document or image, considering gluten intolerance or celiac disease.
    
    Example:
    
    Gluten-free grains - 50%
    Vegetables and fruits - 30%
    Lean protein sources - 20%
    """,
    'Vegetarian or Vegan Diet': """
    Diet Composition Analyzer for Vegetarian or Vegan Diet
    
    Analyze the diet composition based on the provided document or image, tailored for vegetarian or vegan diets.
    
    Example:
    
    Plant-based protein sources - 50%
    Whole grains and legumes - 30%
    Fruits and vegetables - 20%
    """,
    'Food Allergies': """
    Diet Composition Analyzer for Food Allergies
    
    Analyze the diet composition based on the provided document or image, considering food allergies.
    
    Example:
    
    Safe food alternatives - 50%
    Allergen-free substitutes - 30%
    Balanced nutritional choices - 20%
    """,
    'Low Carb/Ketogenic Diet': """
    Diet Composition Analyzer for Low Carb/Ketogenic Diet
    
    Analyze the diet composition based on the provided document or image, tailored for low carb or ketogenic diets.
    
    Example:
    
    Healthy fats - 60%
    Moderate protein - 30%
    Low-carb vegetables - 10%
    """,
    'Athletes or Active Individuals': """
    Diet Composition Analyzer for Athletes or Active Individuals
    
    Analyze the diet composition based on the provided document or image, focusing on the nutritional needs of athletes or active individuals.
    
    Example:
    
    High protein foods - 40%
    Complex carbohydrates - 40%
    Hydration and electrolytes - 20%
    """,
    'Pregnancy or Lactation': """
    Diet Composition Analyzer for Pregnancy or Lactation
    
    Analyze the diet composition based on the provided document or image, tailored for pregnant or lactating women.
    
    Example:
    
    Essential nutrients - 50%
    Calcium and iron-rich foods - 30%
    Hydration and balanced diet - 20%
    """,
    'Heart Disease': """
    Diet Composition Analyzer for Heart Disease
    
    Analyze the diet composition based on the provided document or image, focusing on heart-healthy nutrition.
    
    Example:
    
    Heart-healthy fats - 40%
    Low-sodium foods - 30%
    Lean protein sources - 30%
    """,
    'Gastrointestinal Disorders (e.g., IBS)': """
    Diet Composition Analyzer for Gastrointestinal Disorders (e.g., IBS)
    
    Analyze the diet composition based on the provided document or image, tailored for gastrointestinal disorders.
    
    Example:
    
    Low FODMAP diet - 50%
    Soluble fiber-rich foods - 30%
    Probiotics and gut-friendly choices - 20%
    """,
    'Renal Disease': """
    Diet Composition Analyzer for Renal Disease
    
    Analyze the diet composition based on the provided document or image, tailored for renal disease management.
    
    Example:
    
    Low phosphorus and potassium foods - 50%
    Controlled protein intake - 30%
    Fluid management and hydration - 20%
    """,
    # Add more conditions with specific input prompts here
}

# File upload
uploaded_file = st.file_uploader("Choose a file (PDF, DOC, JPG, JPEG, PNG)...", type=["pdf", "doc", "docx", "jpg", "jpeg", "png"])
image = None

if uploaded_file is not None:
    try:
        image = input_file_setup(uploaded_file)
        if isinstance(image[0], bytes):
            st.error("Error processing document. Please try another file.")
        else:
            st.image(image[0], caption="Uploaded File Preview.", use_column_width=True)
    except Exception as e:
        st.error(f"Error: {e}")

submit = st.button("Tell me about this diet and its composition")

## If submit button is clicked
if submit and image:
    try:
        response = get_gemini_response(image)
        st.subheader("Analysis Results")
        st.write(response)

        # Analyze diet composition and generate pie chart
        food_classes = analyze_diet_and_generate_chart(response)

        # Display tailored recommendations based on patient condition
        if patient_condition in input_prompts:
            st.subheader("Tailored Recommendations")
            st.markdown(f"Recommendations for {patient_condition}:")
            # Display tailored recommendations directly
            recommendations = get_tailored_recommendations(patient_condition)
            st.markdown(recommendations)
        else:
            st.warning("No specific recommendations available for this condition.")

    except Exception as e:
        st.error(f"Error analyzing diet composition: {e}")

from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from PIL import Image 
import google.generativeai as genai #pip install -U google-generativeai
import os 

genai.configure(api_key=os.environ.get('GOOGLE_API_KEY'))



model = genai.GenerativeModel('gemini-pro-vision')

def get_gemini_response(input, image, prompt):
    response = model.generate_content([input, image[0], prompt])
    response.text

def input_image_details(uploaded_file):
    if uploaded_file is not None:
        # Read the file into bytes
        bytes_data = uploaded_file.getvalue()

        image_parts = [
            {
                "mime_type": uploaded_file.type,
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")

# initialize our streamlit app

st.set_page_config(page_title="Image Caption Using Gemini")

st.header("Image Caption Using Gemini")
input = st.text_input("Input Prompt: ", key = "input")
uploaded_file = st.file_uploader("choose an Image of the Invoice....", type=["jpg", "jpeg", "png"])

image = ""
if uploaded_file is not None:
    image=Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

submit=st.button("Tell Me about the Image")


input_prompt="""
You are an expert in understanding invoice. We will upload a image of the invoice
and you  have to understand the invoice based on the uploaded image. 
"""

# if submit button is clicked

if submit:
    image_data=input_image_details(uploaded_file)
    st.text("The Response is:")
    response=get_gemini_response(input_prompt,image_data,input)












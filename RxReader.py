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

st.set_page_config(page_title="prescription interpretation Using Gemini")

st.header("Prescription interpreter")

# input = st.text_input("Input Your Message: ", key = "input")


uploaded_file = st.file_uploader("Choose an Image of the Prescription....", type=["jpg", "jpeg", "png"])

image = ""
if uploaded_file is not None:
    image=Image.open(uploaded_file)
    st.image(image, caption="Uploaded Prescription", use_column_width=True)

submit=st.button("Submit")


input = """ What are the name of the medicine?  and represent as numbered list """

input_prompt="""
You are an expert in understanding prescription written by doctors. We will upload a image of the prescription
written by the doctors and you have to understand the presciption based on the uploaded image of the prescription.
Some prescription has the description when and how the medicine should be taken. You have understand those and 
deliver a reply when we want to know. 
"""





# if submit button is clicked

if submit:
    image_data=input_image_details(uploaded_file)
    st.text("The Response is :")
    response=get_gemini_response(input_prompt,image_data,input)


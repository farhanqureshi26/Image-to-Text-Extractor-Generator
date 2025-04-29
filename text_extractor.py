import streamlit as st
import google.generativeai as genai
import os
import PIL.Image
import pandas as pd

# Set your API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyDXAG4LLRU3YN7PeRcQIaUGXZLsBdZwWGo"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


# custom funtions
model = genai.GenerativeModel("models/gemini-1.5-flash-latest")

def image_to_text(img):
    response = model.generate_content(img)
    return response.text

def image_and_query(img,query):
    response = model.generate_content([query,img])
    return response.text





# app create==============
st.title("Image to Text Extractor & Generator")
st.write("Upload an image and get details about it.")

upload_image = st.file_uploader("Upload an Image", type=['png','jpg','jpeg'])
query = st.text_input("Write a story or blog for this image")

if st.button("Generate"):
    if upload_image and query is not None:
        img = PIL.Image.open(upload_image)
        st.image(img, caption='Uploaded Image', width=300)

        # extract details
        extracted_details = image_to_text(img)
        st.subheader("Extracted Details....")
        st.write(extracted_details)

        # generate details
        generated_details = image_and_query(img,query)
        st.subheader("Generated Details....")
        st.write(generated_details)


        # save to csv files
        data = {"Extracted details":[extracted_details], "Generated Details":[generated_details]}

        df = pd.DataFrame(data)
        csv = df.to_csv(index=False)

        st.download_button(
            label="Download as CSV",
            data = csv,
            file_name="details.csv",
            mime="text/csv"
        )


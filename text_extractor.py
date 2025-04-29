import streamlit as st
import google.generativeai as genai
import os
import PIL.Image
import pandas as pd
from PIL import Image
import imageio
import tempfile
import cv2


# Set your API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyDXAG4LLRU3YN7PeRcQIaUGXZLsBdZwWGo"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


# custom funtions
model = genai.GenerativeModel("models/gemini-1.5-flash-latest")

def image_to_text(img):
    try:
        response = model.generate_content(img)
        #print("Gemini response:", response)
        return response.text or "No meaningful content found"
    except Exception as e:
        return f"Error: {e}"


def image_and_query(img, query):
    try:
        response = model.generate_content([query, img])
        return response.text or "No relevant generation"
    except Exception as e:
        return f"Error: {e}"



# New: process video frame-by-frame
def video_to_text(video_path, frame_interval=60):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    all_text = []

    if not cap.isOpened():
        return "Error: Unable to open video file."

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            text = image_to_text(pil_img)
            all_text.append(f"[Frame {frame_count}]: {text}")
            print(f"Processed frame {frame_count}, got: {text}")
        frame_count += 1

    cap.release()
    if not all_text:
        return "No text could be extracted from any frame."
    return "\n".join(all_text)


def video_and_query(video_path, query, frame_interval=30):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    all_responses = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            result = image_and_query(pil_img, query)
            all_responses.append(f"[Frame {frame_count}]: {result}")
        frame_count += 1
    cap.release()
    return "\n".join(all_responses)

# app create==============
st.title("Image or Video to Text Extractor & Generator")
st.write("Upload an image or video and get details about it.")


upload_image = st.file_uploader("Upload an Image", type=['png','jpg','jpeg'])
upload_video = st.file_uploader("Upload a Video", type=['mp4','avi','m4v','.3gp'])
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
    if upload_video and query:
        # Save uploaded video to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(upload_video.read())
            temp_video_path = tmp_file.name

        # Show video
        st.video(upload_video)

        # Extract details
        extracted_details = video_to_text(temp_video_path)
        st.write("Temporary video path:", temp_video_path)
        st.write("Query:", query)

        if not extracted_details.strip():
            st.warning(
                "No meaningful content was extracted from the video. Try a different video or reduce the frame interval.")
        else:
            st.subheader("Extracted Details")
            st.write(extracted_details)

        # Generate details
        generated_details = video_and_query(temp_video_path, query)
        st.subheader("Generated Details")
        st.write(generated_details)

        # Save results to CSV
        data = {
            "Extracted details": [extracted_details],
            "Generated Details": [generated_details]
        }
        df = pd.DataFrame(data)
        csv = df.to_csv(index=False)

        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name="details.csv",
            mime="text/csv"
        )






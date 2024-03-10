import streamlit as st
from PIL import Image

def main():
    st.title("Image File Uploader")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

        if st.button("Submit"):
            process_image(uploaded_file)

def process_image(uploaded_file):
    image = Image.open(uploaded_file)

    # Check if the file is an image
    if image.format not in ["JPEG", "PNG"]:
        st.error("Uploaded file is not a valid image format. Please upload a JPEG or PNG file.")
        return

    # Additional processing logic can be added here

    st.success("Image processing completed successfully!")

if __name__ == "__main__":
    main()

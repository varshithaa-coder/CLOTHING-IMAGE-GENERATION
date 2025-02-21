import torch
import streamlit as st
from diffusers import StableDiffusionPipeline
from PIL import Image

# Model Selection
MODEL_ID = "runwayml/stable-diffusion-v1-5"

def load_pipeline():
    """Load the Stable Diffusion model pipeline."""
    pipeline = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    if torch.cuda.is_available():
        pipeline = pipeline.to("cuda")
    return pipeline

# Load pipeline once and cache it
@st.cache_resource
def get_pipeline():
    return load_pipeline()

def generate_image(pipeline, prompt):
    """Generate an image from a text prompt."""
    image = pipeline(prompt).images[0]
    return image

# Streamlit UI
st.title("AI Clothing Generator")
st.write("Describe your clothing design, and AI will generate an image for you!")

user_prompt = st.text_input("Enter your clothing description:", "A stylish red jacket with gold embroidery")

generate_button = st.button("Generate Image")

if generate_button and user_prompt:
    with st.spinner("Generating image... Please wait."):
        pipeline = get_pipeline()
        generated_image = generate_image(pipeline, user_prompt)
        
        st.image(generated_image, caption="Generated Clothing Design", use_column_width=True)
        
        # Save and allow download
        image_path = "generated_clothing.png"
        generated_image.save(image_path)
        
        with open(image_path, "rb") as file:
            st.download_button(
                label="Download Image",
                data=file,
                file_name="generated_clothing.png",
                mime="image/png"
            )

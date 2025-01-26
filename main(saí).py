import streamlit as st
import torch
import random
import sys
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline
from PIL import Image
from io import BytesIO

def generate_image(prompt, num_inference_steps, guidance_scale, eta):
    # Device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load pipeline directly from Stability AI model
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", 
        torch_dtype=torch.float16, 
        variant="fp16"
    ).to(device)

    # Generate seed
    seed = random.randint(0, sys.maxsize)

    # Generate image
    generator = torch.Generator(device).manual_seed(seed)
    images = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator
    ).images

    return images[0], seed

def main():
    st.title("🎨 AI Image Generator")

    # Sidebar for configuration
    st.sidebar.header("Image Generation Settings")
    
    # Prompt input
    prompt = st.sidebar.text_area("Enter Prompt", 
        "A female long brown hair musician playing a piano in an old retro studio with a group of 3 young boys ages from 5 to 11 singing dressed with barroque style jackets")
    
    # Inference steps
    num_inference_steps = st.sidebar.slider("Inference Steps", 8, 50, 30)
    
    # Guidance scale
    guidance_scale = st.sidebar.slider("Guidance Scale", 1.0, 20.0, 7.5, 0.1)

    # Generate button
    if st.sidebar.button("Generate Image"):
        with st.spinner("Generating image..."):
            try:
                image, seed = generate_image(prompt, num_inference_steps, guidance_scale, 0.5)
                
                # Display results
                st.image(image, caption=f"Generated Image (Seed: {seed})")
                
                # Download button
                img_byte_arr = BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                st.download_button(
                    label="Download Image",
                    data=img_byte_arr,
                    file_name="generated_image.png",
                    mime="image/png"
                )
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

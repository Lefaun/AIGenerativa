import streamlit as st
import torch
import random
import sys
from diffusers import DiffusionPipeline, TCDScheduler
from huggingface_hub import hf_hub_download
from PIL import Image

def generate_image(prompt, num_inference_steps, guidance_scale, eta):
    # Device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model and LoRA configuration
    base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    repo_name = "ByteDance/Hyper-SD"
    plural = "s" if num_inference_steps > 1 else ""
    ckpt_name = f"Hyper-SDXL-{num_inference_steps}step{plural}-CFG-lora.safetensors"

    # Load pipeline
    pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, variant="fp16").to(device)
    pipe.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
    pipe.fuse_lora()
    pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

    # Generate seed
    seed = random.randint(0, sys.maxsize)

    # Generate image
    generator = torch.Generator(device).manual_seed(seed)
    images = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        eta=eta,
        generator=generator
    ).images

    return images[0], seed

def main():
    st.title("🎨 Hyper-SD Image Generator")

    # Sidebar for configuration
    st.sidebar.header("Image Generation Settings")
    
    # Prompt input
    prompt = st.sidebar.text_area("Enter Prompt", 
        "A female long brown hair musician playing a piano in an old retro studio with a group of 3 young boys ages from 5 to 11 singing dressed with barroque style jackets")
    
    # Inference steps
    num_inference_steps = st.sidebar.slider("Inference Steps", 8, 12, 12)
    
    # Guidance scale
    guidance_scale = st.sidebar.slider("Guidance Scale", 1.0, 10.0, 5.0, 0.1)
    
    # Eta value
    eta = st.sidebar.slider("Eta Value", 0.0, 1.0, 0.5, 0.1)

    # Generate button
    if st.sidebar.button("Generate Image"):
        with st.spinner("Generating image..."):
            image, seed = generate_image(prompt, num_inference_steps, guidance_scale, eta)
            
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

if __name__ == "__main__":
    main()

import streamlit as st
import torch
from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler

@st.cache_resource
def carregar_modelo():
    modelo = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to("cuda")
    modelo.scheduler = DPMSolverMultistepScheduler.from_config(modelo.scheduler.config)
    return modelo

def gerar_imagem(modelo, prompt):
    return modelo(
        prompt=prompt, 
        num_inference_steps=20,
        guidance_scale=7.5
    ).images[0]

def main():
    st.title("🚀 Gerador Rápido de Imagens")
    modelo = carregar_modelo()
    
    prompt = st.text_input("Descrição da imagem:")
    if st.button("Gerar Rápido"):
        try:
            imagem = gerar_imagem(modelo, prompt)
            st.image(imagem)
        except Exception as e:
            st.error(f"Erro: {e}")

if __name__ == "__main__":
    main()

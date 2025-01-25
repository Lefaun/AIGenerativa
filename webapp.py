import streamlit as st
import torch
from diffusers import StableDiffusionXLPipeline

def gerar_imagem(prompt):
    modelo = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0"
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    imagem = modelo(prompt=prompt).images[0]
    return imagem

def main():
    st.title("🎨 Gerador de Imagens")
    prompt = st.text_input("Digite sua descrição em português:")
    
    if st.button("Gerar Imagem"):
        imagem = gerar_imagem(prompt)
        st.image(imagem)
        imagem.save("imagem_gerada.png")
        st.download_button("Baixar Imagem", data=open("imagem_gerada.png", "rb"))

if __name__ == "__main__":
    main()

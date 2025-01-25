import streamlit as st
import torch
from diffusers import StableDiffusionPipeline

st.title("Gerador de Imagens com Stable Diffusion")

# Carregar o modelo Stable Diffusion
model_id = "CompVis/stable-diffusion-v1-4"  # Você pode usar outros modelos disponíveis
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True)
# pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")  # Mova o modelo para a GPU, se disponível

def gerar_imagem(descricao, num_imagens=1):
    # Gerar imagens a partir da descrição
    imagens = pipe(descricao, num_images=num_imagens).images
    return imagens

# Interface do usuário
descricao = st.text_input("Descreva a imagem que você deseja gerar:")
if st.button("Gerar Imagem"):
    if descricao:
        imagens_geradas = gerar_imagem(descricao)

        # Salvar e mostrar as imagens geradas
        for i, img in enumerate(imagens_geradas):
            img.save(f"imagem_gerada_{i}.png")
            st.image(img, caption=f"Imagem gerada {i+1}", use_column_width=True)
    else:
        st.warning("Por favor, insira uma descrição para gerar a imagem.")

import streamlit as st
import torch
from diffusers import StableDiffusionPipeline


st.title("Gerador de Imagens com Stable Diffusion")

# Carregar o modelo Stable Diffusion
model_id = "CompVis/stable-diffusion-v1-4"  # Você pode usar outros modelos disponíveis
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")  # Mova o modelo para a GPU, se disponível

def gerar_imagem(descricao, num_imagens=1):
    # Gerar imagens a partir da descrição
    imagens = pipe(descricao, num_images=num_imagens).images
    return imagens

# Exemplo de uso
if __name__ == "__main__":
    descricao = st.text_input("Um castelo mágico em uma floresta encantada")
    imagens_geradas = gerar_imagem(descricao)

    # Salvar e mostrar as imagens geradas
    for i, img in enumerate(imagens_geradas):
        img.save(f"imagem_gerada_{i}.png")
        st.image=img.show()

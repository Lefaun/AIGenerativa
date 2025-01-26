import streamlit as st
import torch
from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler

@st.cache_resource
def carregar_modelo():
    try:
        if torch.cuda.is_available():
            modelo = AutoPipelineForText2Image.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True
            ).to("cuda")
        else:
            modelo = AutoPipelineForText2Image.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                use_safetensors=True
            ).to("cpu")
        
        modelo.scheduler = DPMSolverMultistepScheduler.from_config(modelo.scheduler.config)
        return modelo
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None

def gerar_imagem(modelo, prompt):
    try:
        return modelo(
            prompt=prompt, 
            num_inference_steps=20,
            guidance_scale=7.5
        ).images[0]
    except Exception as e:
        raise RuntimeError(f"Erro ao gerar a imagem: {e}")

def main():
    st.title("🚀 Gerador Rápido de Imagens")
    modelo = carregar_modelo()
    
    if modelo is None:
        st.error("Falha ao carregar o modelo. Verifique o log de erros acima.")
        return

    prompt = st.text_input("Descrição da imagem:")
    if st.button("Gerar Rápido"):
        if not prompt.strip():
            st.warning("Por favor, insira uma descrição válida para a imagem.")
            return
        
        with st.spinner("Gerando imagem, por favor aguarde..."):
            try:
                imagem = gerar_imagem(modelo, prompt)
                st.image(imagem, caption="Imagem Gerada", use_column_width=True)
            except RuntimeError as e:
                st.error(e)

if __name__ == "__main__":
    main()

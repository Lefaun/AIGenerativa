import torch
from diffusers import DPMSolverMultistepScheduler

modelo.scheduler = DPMSolverMultistepScheduler.from_config(modelo.scheduler.config)



@st.cache_resource
def carregar_modelo():
    return AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,  # Use half precision
        variant="fp16",
        use_safetensors=True
    ).to("cuda")  # Ensure GPU usage
    
    imagem = modelo(
    prompt=prompt, 
    num_inference_steps=20,  # Reduce from default 50
    guidance_scale=7.5
)
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

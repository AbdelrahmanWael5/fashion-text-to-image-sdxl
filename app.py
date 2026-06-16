import streamlit as st
import time

from inference import ClothingGenerator


@st.cache_resource
def load_model():

    return ClothingGenerator(
        model_path="stabilityai/stable-diffusion-xl-base-1.0",
        checkpoint_path="best_model_5.pth",
    )

generator = load_model()

st.set_page_config(
    page_title="Fashion Text-to-Image",
    page_icon="👔",
)

st.title("👔 Fashion Text-to-Image Generator")

st.write(
    "Custom SDXL model fine-tuned on fashion datasets."
)

prompt = st.text_area(
    "Prompt",
    value="A luxury black leather jacket"
)

if st.button("Generate Image"):

    start = time.time()

    image = generator.generate_image(
        prompt=prompt,
    )

    elapsed = round(time.time() - start, 2)

    st.image(
        image,
        caption=f"Generated in {elapsed} seconds",
        use_container_width=True,
    )
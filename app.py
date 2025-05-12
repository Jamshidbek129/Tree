from fastai.vision.all import *
import streamlit as st
from pathlib import Path
import plotly.express as px
import pandas as pd

st.title("Daraxtlar va Gullar")

# Model yo‘li
model_path = Path("model.pkl")
if model_path.exists():
    model = load_learner(model_path)
else:
    st.error("Model fayli topilmadi!")

fayl = st.file_uploader("Rasm yuklash", type=['png', 'jpg', 'jpeg'])

if fayl:
    rasm = PILImage.create(fayl)
    bashorat, id, ehtimollik = model.predict(rasm)

    st.image(fayl)
    st.success(f"Bashorat: {bashorat}")
    st.info(f"Ehtimollik: {ehtimollik[id]*100:.1f}%")

    df = pd.DataFrame({'Sinflar': model.dls.vocab, 'Ehtimollik (%)': ehtimollik*100})
    fig = px.bar(df, x='Sinflar', y='Ehtimollik (%)', title='Ehtimolliklar')
    st.plotly_chart(fig)

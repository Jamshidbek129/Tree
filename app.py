from fastai.vision.all import *
import streamlit as st
import pathlib
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
import platform

if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath

st.text("Assalomu aleykum")
st.title("Daraxtlar va Gullar")

fayl=st.file_uploader("Rasm yuklash",  type=['png', 'gif', 'svg', 'jpeg'])
if fayl:
  rasm=PILImage.create(fayl)

  model=load_learner('model (1).pkl')

  bashorat, id, ehtimollik=model.predict(rasm)

  st.image(rasm)
  st.success(f"Bashorat: {bashorat}")
  st.info(f"Ehtimollik: {ehtimollik[id]*100:.1f}%")
  natija = pd.DataFrame({
    'Sinflar': model.dls.vocab,
    'Ehtimollik (%)': ehtimollik*100
  })

  fig = px.bar(natija, x='Sinflar', y='Ehtimollik (%)', title="Bashorat ehtimolliklari")
  fig=px.bar(y=ehtimollik*100, x=model.dls.vocab)
  st.plotly_chart(fig)

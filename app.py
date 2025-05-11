from fastai.vision.all import *
import streamlit as st
import pathlib
import plotly.express as px
import matplotlib.pyplot as plt
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

st.text("Assalomu aleykum")
st.title("Daraxtlar va Gullar")

fayl=st.file_uploader("Rasm yuklash",  type=['png', 'gif', 'svg', 'jpeg'])
if fayl:
  rasm=PILImage.create(fayl)

  model=load_learner('model (1).pkl')

  bashorat, id, ehtimollik=model.predict(rasm)

  st.image(fayl)
  st.success(f"Bashorat: {bashorat}")
  st.info(f"Ehtimollik: {ehtimollik[id]*100:.1f}%")

  fig=px.bar(y=ehtimollik*100, x=model.dls.vocab)
  st.plotly_chart(fig)
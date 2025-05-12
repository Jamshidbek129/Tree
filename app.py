from fastai.vision.all import *
import streamlit as st
import plotly.express as px
import pathlib
import platform

plt = platform.system()
st.write(plt)  # just for debugging
if plt == 'Linux':
    pathlib.PosixPath = pathlib.WindowsPath  # or maybe the other way round

st.text("Assalomu aleykum")
st.title("Daraxtlar va Gullar")

fayl=st.file_uploader("Rasm yuklash",  type=['png', 'gif', 'svg', 'jpeg'])
# if fayl:
#   rasm=PILImage.create(fayl)

#   model=load_learner('model.pkl')

#   bashorat, id, ehtimollik=model.predict(rasm)

#   st.image(fayl)
#   st.success(f"Bashorat: {bashorat}")
#   st.info(f"Ehtimollik: {ehtimollik[id]*100:.1f}%")

#   fig=px.bar(y=ehtimollik*100, x=model.dls.vocab)
#   st.plotly_chart(fig)
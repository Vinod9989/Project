

from img import image_classification
import streamlit as st
import keras
from PIL import Image, ImageOps
import numpy as np
import pickle

st.title("Glaucoma Classification")
st.header("Glaucoma identification using machine learning")
st.text("Upload an image")
uploaded_file = st.file_uploader("Choose a Gluacoma image ...", type="jpg")



if uploaded_file is not None:
	image = Image.open(uploaded_file).convert('L')
	image_data = np.asarray(image)
	st.image(image, caption='Uploaded image.', use_column_width=True)
	st.write("")
	st.write("Classifying...")
	label = image_classification(image_data)


	if (label == "Glaucoma_Negative"):
		st.write("It is a Glaucoma_Negative images")


	else:
		st.write("It is Glaucoma_Postive images")
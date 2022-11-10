import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import pickle
import cv2
from skimage.transform import resize
from skimage.feature import hog




model = pickle.load(open('lr.p','rb'))


def image_classification(img):
    # Load the model   
    img_resize=cv2.resize(img,(224,224))
    img = hog(img_resize,orientations=9,pixels_per_cell=(8,8),visualize=False,cells_per_block=(3,3))
    img_data = np.float32(img)
    img=(np.expand_dims(img_data,0))
    probability=model.predict(img)

    return probability # return position of the highest probability



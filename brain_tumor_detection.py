import streamlit as st

from PIL import Image
import streamlit as st
import pandas as pd
from tensorflow import keras
import numpy as np
import cv2


st.header("Brain Tumor Classification")
st.text("Upload a brain MRI Image for classifying as tumor or no-tumor")

#@st.cache(allow_output_mutation=True)
@st.cache_resource
def load_model():
    model = keras.models.load_model('inception_v3_model.h5')
    return model

model= load_model()

def preprocess_predict(image):
    print((image))

    images= cv2.resize(image,(224,224))/255
    img_array= np.array(images)
    img_array= img_array[np.newaxis,...]
    
    #print(img_array)
    print(img_array.shape)
    
    
    result = model.predict(img_array)
    if np.argmax(result,axis=1)==0:
        output= 'MRI image is 100% Normal'
    else: 
        output= 'MRI image is suffering from Brain Tumor'

    s='predicted  {} with probability : {}'.format(output,np.max(result,axis=1))
    return s


img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if img_file_buffer is not None:
    image = Image.open(img_file_buffer).convert('RGB')
    image= np.array(image)   
    pred=preprocess_predict(image)
    if st.button("Predict"):
        st.success(pred)
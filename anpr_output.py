# %%
# remove warning message
from os.path import splitext, basename
import streamlit as st
import glob
from PIL import Image, ImageOps
from sklearn.preprocessing import LabelEncoder
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing import image
from keras.models import model_from_json
from local_utils import detect_lp
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import numpy as np
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# required library
#from keras.preprocessing.image import load_img, img_to_array

# %%


def anpr(plate_img):
    img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
    if (len(img)):
        # plate_image = cv2.convertScaleAbs(img, alpha=(255.0))
        gray = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(np.float32(gray), (7, 7), 0)
        binary = cv2.threshold(np.float32(blur), 180, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)

    # Create sort_contours() function to grab the contour of each digit from left to right
    def sort_contours(cnts, reverse=False):
        i = 0
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                            key=lambda b: b[1][i], reverse=reverse))
        return cnts

    cont, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    test_roi = img.copy()
    crop_characters = []
    digit_w, digit_h = 180, 180

    for c in sort_contours(cont):
        (x, y, w, h) = cv2.boundingRect(c)
        ratio = h/w
        if 1 <= ratio <= 5:  # Only select contour with defined ratio
            if h/img.shape[0] >= 0.3:
                cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
                curr_num = thre_mor[y:y+h, x:x+w]
                curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                _, curr_num = cv2.threshold(
                    curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                crop_characters.append(curr_num)

    # loading model 2.0
    model = keras.models.load_model('Model 3.0.h5')
    out = []
    for i in range(len(crop_characters)):
        x = crop_characters[i]
        x = np.reshape(x, (180, 180, 1))
        x = np.concatenate([x, x, x], axis=2)
        x = np.expand_dims(x, axis=0)
        x = np.concatenate([x for _ in range(32)], axis=0)
        p1 = model.predict(x)

        c = np.argmax(p1)
        out.append(c)

    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
               'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    txt = ""
    for i in (out):
        if(i < 36):
            txt = txt+classes[i]
        else:
            txt = txt+"-"
    return txt


uploaded_file = st.file_uploader("Choose lisence plate Image...", type="png")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.write(np.shape(image))
    st.image(image, caption='Uploaded file', use_column_width=True)
    output = anpr(image)
    st.write("Classifying...")
    st.write(output)
# img = cv2.imread('image.png')
#img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#cv2.imwrite('opncv_sample.png', img)
# anpr(img)

# %%
# import tensorflow as tf
# import cv2
# from PIL import Image, ImageOps
# import numpy as np
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import PIL
# import tensorflow as tf

# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.models import Sequential

# st.title("Cat-Dog Classification")
# st.header("Please input an image to be classified:")
# st.text("Created by Saksham Gulati")

# @st.cache(allow_output_mutation=True)

# def teachable_machine_classification(img, weights_file):
#     # Load the model
#     model = keras.models.load_model(weights_file)

#     # Create the array of the right shape to feed into the keras model
#     data = np.ndarray(shape=(1, 200, 200, 3), dtype=np.float32)
#     image = img
#     #image sizing
#     size = (200, 200)
#     image = ImageOps.fit(image, size, Image.ANTIALIAS)

#     #turn the image into a numpy array
#     image_array = np.asarray(image)
#     # Normalize the image
#     normalized_image_array = (image_array.astype(np.float32) / 255)

#     # Load the image into the array
#     data[0] = normalized_image_array

#     # run the inference
#     prediction_percentage = model.predict(data)
#     prediction=prediction_percentage.round()

#     return  prediction,prediction_percentage

# label,perc = teachable_machine_classification(image, 'catdog.h5')
# if label == 1:
#     st.write("Its a Dog, confidence level:",perc)
# else:
#     st.write("Its a Cat, confidence level:",1-perc)

# %%

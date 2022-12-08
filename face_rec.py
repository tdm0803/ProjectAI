import cv2
import streamlit as st
import os
import numpy as np
import pandas as pd
import face_recognition
import cv2
from google_images_download import google_images_download 
import sys
from PIL import Image
from os import listdir

# CONSTANTS
PATH_DATA = 'data/DB.csv'
COLOR_DARK = (0, 0, 153)
COLOR_WHITE = (255, 255, 255)
COLS_INFO = ['name']
COLS_ENCODE = [f'v{i}' for i in range(128)]


def init_data(data_path=PATH_DATA):
    if os.path.isfile(data_path):
        return pd.read_csv(data_path)
    else:
        return pd.DataFrame(columns=COLS_INFO + COLS_ENCODE)

# convert image from opened file to np.array


def byte_to_array(image_in_byte):
    return cv2.imdecode(
        np.frombuffer(image_in_byte.read(), np.uint8),
        cv2.IMREAD_COLOR
    )

# convert opencv BRG to regular RGB mode


def BGR_to_RGB(image_in_array):
    return cv2.cvtColor(image_in_array, cv2.COLOR_BGR2RGB)

# convert face distance to similirity likelyhood



def face_distance_to_conf(face_distance, face_match_threshold=0.4):
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * np.power((linear_val - 0.5) * 2, 0.2))

def extract_image(image):
    img1 = Image.open(image)            #open the image
    img1 = img1.convert('RGB')          #convert the image to RGB format
   
    return img1

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    # st.set_option("deprecation.showfileUploaderEncoding", False)
    st.image("/mnt/hdd2T/tham/projectAI/GUI/ProjectAI_Streamlit/download.png"
        
    )
    # st.sidebar.image("/mnt/hdd2T/tham/projectAI/GUI/ProjectAI_Streamlit/download.png", use_column_width=True)
    # title area
    st.markdown("""
    # Personal Information Detection and Deletion System
    ## My-Tham Dinh - Project AI - 2022
    """)
    col1,col2,col3,col4 = st.columns([1,2,1,1])
    # displays a file uploader widget and return to BytesIO
    image_byte = col1.file_uploader(
        label="Select a picture contains faces:", type=['jpg', 'png','jpeg']
    )
    # detect faces in the loaded image
    max_faces = 0
    rois = []  # region of interests (arrays of face areas)
    if image_byte is not None:
        image_array = byte_to_array(image_byte)
        face_locations = face_recognition.face_locations(image_array,model="cnn")
        for idx, (top, right, bottom, left) in enumerate(face_locations):
            # save face region of interest to list
            rois.append(image_array[top:bottom, left:right].copy())

            # Draw a box around the face and lable it
            cv2.rectangle(image_array, (left, top),
                          (right, bottom), COLOR_DARK, 2)
            cv2.rectangle(
                image_array, (left, bottom + 35),
                (right, bottom), COLOR_DARK, cv2.FILLED
            )
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(
                image_array, f"#{idx}", (left + 5, bottom + 25),
                font, .55, COLOR_WHITE, 1
            )

        col1.image(BGR_to_RGB(image_array), width=300)
        max_faces = len(face_locations)

    if max_faces > 0:
        # select interested face in picture
        face_idx = col1.selectbox("Select face#", range(max_faces))
        roi = rois[face_idx]
        col2.image(BGR_to_RGB(roi), width=min(roi.shape[0], 300))

        # initial database for known faces
        DB = init_data()
        face_encodings = DB[COLS_ENCODE].values
        dataframe = DB[COLS_INFO]

        # compare roi to known faces, show distances and similarities
        face_to_compare = face_recognition.face_encodings(roi)[0]
        dataframe['distance'] = face_recognition.face_distance(
            face_encodings, face_to_compare)


        dataframe['similarity'] = dataframe.distance.apply(
             lambda distance: f"{face_distance_to_conf(distance):0.2%}")

        # if dataframe.sort_values("distance").iloc[:].iloc[0][1] <=0.5:
        col2.dataframe(
            dataframe.sort_values("distance")[dataframe.distance <=0.4])

            
        a=dataframe.sort_values("distance")[dataframe['distance']<=0.4].loc[:, dataframe.columns =='name'].values.tolist()
        
        directory = '/mnt/hdd2T/tham/projectAI/GUI/ProjectAI_Streamlit/crawl/downloads/'
        
        
        for idx in range(len(a)):
            images = directory+a[idx][0]
            list_image = extract_image(images)
            
            col3.image(list_image, width=150)
            
            new_a = a[idx][0].split('/')
            print(new_a)
            
            image_name = new_a[1][len(new_a[1].split('.')[0])+1:]
            print(image_name)
            with open(directory+new_a[0]+'.txt', 'r' ) as f:
                lines = f.readlines()
                #print(lines[1])
                for line in lines:
                    if image_name in line:
                        print(line)
                        
                        col4.markdown(f" {line}")


                        
            #             col4.markdown(f" {line}")       


        # add roi to known database
        if st.checkbox('add it to knonwn faces'):
            face_name = st.text_input('Name:', '')
            
            if st.button('add'):
                encoding = face_to_compare.tolist()
                DB.loc[len(DB)] = [face_name] + encoding
                DB.to_csv(PATH_DATA, index=False)
    else:
        st.write('No human face detected.')

    

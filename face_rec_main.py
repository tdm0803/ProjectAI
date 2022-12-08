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
from retinaface import RetinaFace
from arcface import ArcFace
from os import listdir
import os
import torch.nn as nn
import torch
from numpy import asarray
import random as rand

# import requests, sys
# from time import time as timer
# from multiprocessing.dummy import Pool as ThreadPool
 

 
# def redpage(url):
 
#     try:
 
#         req_check = requests.get(url, verify=False)
 
#         if 'malicious words' in req_check.content:
#             print ('[Your Site Is Detect Malicious Site]   ===>   '+url)
#         else:
#             print ('[Your Site Clean]   ===>   '+url)
 
#     except:
#         pass
 
# def checking(url):
 
#     try:
#         redpage(url)
#     except:
#         pass

 


# CONSTANTS
PATH_DATA = 'data/DB.csv'
COLOR_DARK = (0, 0, 153)
COLOR_WHITE = (255, 255, 255)
COLS_INFO = ['name']
COLS_ENCODE = [f'v{i}' for i in range(512)]


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
    # Personal Information Detection System
    ## My-Tham Dinh - Project AI - 2022
    """)
    col1,col2,col3,col4 = st.columns([1,2,1,1])
    # displays a file uploader widget and return to BytesIO
    image_byte = col1.file_uploader(
        label="Select a picture contains faces:", type=['jpg', 'png','jpeg','webp']
    )
    # detect faces in the loaded image
    max_faces = 0
    rois = []  # region of interests (arrays of face areas)
    if image_byte is not None:
        image_b = extract_image(image_byte)
        image_array = asarray(image_b)   
        detector = RetinaFace()                 #assign the MTCNN detector
        f = detector.predict(image_array)
        max_faces =0
        rois = []
        for idx in range(len(f)):
            
            x1,y1,x2,y2 = f[idx]['x1'],f[idx]['y1'],f[idx]['x2'],f[idx]['y2']
            
            # x1, y1 = abs(x1), abs(y1)
            # x2 = abs(x1+w)
            # y2 = abs(y1+h)
                
            roi_face = image_array[y1:y2,x1:x2]
            # image1 = Image.fromarray(store_face,'RGB')    #convert the numpy array to object
            # image1 = image1.resize((112,112))             #resize the image
            # face_array = asarray(image1) 
            cv2.rectangle(image_array, (x1, y1),
                        (x2, y2), COLOR_DARK, 2)
            cv2.rectangle(
                image_array, (x1, y2 + 35),
                (x2, y2), COLOR_DARK, cv2.FILLED
            )
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(
                image_array, f"#{idx}", (x1 + 5, y2 + 25),
                font, .55, COLOR_WHITE, 1
            )
            rois.append(roi_face)
        col1.image((image_array), width=300)     
        max_faces = len(f)

        if max_faces > 0:
            # select interested face in picture
            #for face_idx in range(max_faces):
            face_idx = col1.selectbox("Select face#", range(max_faces))
            roi = rois[face_idx]
            col2.image((roi), width=min(roi.shape[0], 300))

            DB = init_data()
            face_encodings = DB[COLS_ENCODE].values
            dataframe = DB[COLS_INFO]

            face_rec = ArcFace.ArcFace()
            #print(face_rec)
            face_to_compare = face_rec.calc_emb(roi)
            
            # ln1 = nn.Linear(512,128)
            # face_to_compare_128=ln1(torch.tensor(face_to_compare))
            new_encoding=face_to_compare.tolist()
            print(len(new_encoding))
            faces_dis=[]
            for i in range(len(dataframe)):
                #print(len(DB[COLS_ENCODE].loc[i].values))
                #print(face_rec.get_distance_embeddings(DB[COLS_ENCODE].loc[i].values.tolist(),new_encoding))
                faces_dis.append(face_rec.get_distance_embeddings(DB[COLS_ENCODE].loc[i].values.tolist(),new_encoding))
            dataframe['distance'] = faces_dis
            

            # dataframe['distance'] = face_rec.get_distance_embeddings(
            # face_encodings,new_encoding)
            # dataframe['similarity'] = dataframe.distance.apply(
            #  lambda distance: f"{face_distance_to_conf(distance):0.1%}")

            col2.dataframe(
               dataframe.sort_values("distance")[dataframe.distance <=1.15])
            # # a=dataframe.sort_values("distance")[dataframe['distance']<=0.4].loc[:, dataframe.columns =='name'].values.tolist()
            

            a=dataframe.sort_values("distance")[dataframe['distance']<=1.15].loc[:, dataframe.columns =='name'].values.tolist()
        
            directory = '/mnt/hdd2T/tham/projectAI/GUI/ProjectAI_Streamlit/crawl/downloads/'
            print(len(a))
        
            for idx2 in range(len(a)):
                images = directory+a[idx2][0]
                
                list_image = extract_image(images)
                
                col3.image(list_image, width=150)
                
                new_a = a[idx2][0].split('/')
                
                image_name = new_a[1][len(new_a[1].split('.')[0])+1:]
                with open(directory+new_a[0]+'.txt', 'r' ) as f:
                    lines = f.readlines()
                    #print(lines[1])
                    for line in lines:
                        if image_name.lower() in line.lower():
                            #print(line)
                            col4.markdown(f" {line}")
                            
                            # woh = (line)
                            # req_check = requests.get(woh, verify=False)
 
                            # if 'malicious words' in req_check.content:
                            #     print ('[Your Site Is Detect Malicious Site]   ===>   '+woh)
                            # else:
                            #     print ('[Your Site Clean]   ===>   '+woh)
                            # try:
                            #     start = timer()
                            #     pp = ThreadPool(25)
                            #     pr = pp.map(redpage(woh), woh)
                            #     col4.mardown((f" {pr}"))
                                
                            # except:
                            #     pass

                            
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

    




            # DB.loc[len(DB)] = [name] + new_encoding
            # DB.to_csv(PATH_DATA, index=False)
                #print(face_idx)
                # try:
                
                # st.image(BGR_to_RGB(roi), width=min(roi.shape[0], 400))

                # initial database for known faces
                # DB = init_data()
                # face_encodings = DB[COLS_ENCODE].values
                # #dataframe = DB[COLS_INFO]
                # face_rec = ArcFace.ArcFace()
                # #print(face_rec)
                # face_to_compare = face_rec.calc_emb(roi)
                # ln1 = nn.Linear(512,128)
                # face_to_compare_128=ln1(torch.tensor(face_to_compare))
                # new_encoding=face_to_compare_128.tolist()
                
                #face_to_compare = ArcFace.face_encodings(roi)[0]
                # encoding =  nn.ModuleList(face_to_compare_128 )
                # print((encoding))
                # new_encoding = pd.DataFrame(COLS_ENCODE,columns=COLS_INFO )
                # new_encoding.append(pd.Series(encoding, index=new_encoding.columns[:len(encoding)]), ignore_index=True) 
                # DB.loc[len(DB)] = [name] + new_encoding
                # DB.to_csv(PATH_DATA, index=False)


        # image_array = byte_to_array(image_byte)
        # face_locations = face_recognition.face_locations(image_array,model="cnn")
        # for idx, (top, right, bottom, left) in enumerate(face_locations):
        #     # save face region of interest to list
        #     rois.append(image_array[top:bottom, left:right].copy())

        #     # Draw a box around the face and lable it
        #     cv2.rectangle(image_array, (left, top),
        #                   (right, bottom), COLOR_DARK, 2)
        #     cv2.rectangle(
        #         image_array, (left, bottom + 35),
        #         (right, bottom), COLOR_DARK, cv2.FILLED
        #     )
        #     font = cv2.FONT_HERSHEY_DUPLEX
        #     cv2.putText(
        #         image_array, f"#{idx}", (left + 5, bottom + 25),
        #         font, .55, COLOR_WHITE, 1
        #     )

        # col1.image(BGR_to_RGB(image_array), width=300)
        # max_faces = len(face_locations)

    
                # if max_faces > 0:
                #     # select interested face in picture
                #     for face_idx in range(max_faces):
                #         #print(face_idx)
                #         # try:
                #         roi = rois[face_idx]
                #         # st.image(BGR_to_RGB(roi), width=min(roi.shape[0], 400))

                #         # initial database for known faces
                #         DB = init_data()
                #         face_encodings = DB[COLS_ENCODE].values
                #         #dataframe = DB[COLS_INFO]
                #         face_rec = ArcFace.ArcFace()
                #         #print(face_rec)
                #         face_to_compare = face_rec.calc_emb(roi)
                #         ln1 = nn.Linear(512,128)
                #         face_to_compare_128=ln1(torch.tensor(face_to_compare))
                #         new_encoding=face_to_compare_128.tolist()
                        
                #         #face_to_compare = ArcFace.face_encodings(roi)[0]
                #         # encoding =  nn.ModuleList(face_to_compare_128 )
                #         # print((encoding))
                #         # new_encoding = pd.DataFrame(COLS_ENCODE,columns=COLS_INFO )
                #         # new_encoding.append(pd.Series(encoding, index=new_encoding.columns[:len(encoding)]), ignore_index=True) 
                #         DB.loc[len(DB)] = [name] + new_encoding
                #         DB.to_csv(PATH_DATA, index=False)
        # select interested face in picture
        # face_idx = col1.selectbox("Select face#", range(max_faces))
        # roi = rois[face_idx]
        # col2.image(BGR_to_RGB(roi), width=min(roi.shape[0], 300))

        # # initial database for known faces
        # DB = init_data()
        # face_encodings = DB[COLS_ENCODE].values
        # dataframe = DB[COLS_INFO]

        # # compare roi to known faces, show distances and similarities
        # face_to_compare = face_recognition.face_encodings(roi)[0]
        # dataframe['distance'] = face_recognition.face_distance(
        #     face_encodings, face_to_compare)


        # dataframe['similarity'] = dataframe.distance.apply(
        #      lambda distance: f"{face_distance_to_conf(distance):0.1%}")

        # # if dataframe.sort_values("distance").iloc[:].iloc[0][1] <=0.5:
        # col2.dataframe(
        #     dataframe.sort_values("distance")[dataframe.distance <=0.4])
        # a=dataframe.sort_values("distance")[dataframe['distance']<=0.4].loc[:, dataframe.columns =='name'].values.tolist()
        
        # directory = '/mnt/hdd2T/tham/projectAI/GUI/ProjectAI_Streamlit/crawl/downloads/'
        
        
        # for idx in range(len(a)):
        #     images = directory+a[idx][0]
        #     list_image = extract_image(images)
            
        #     col3.image(list_image, width=150)
            
        #     new_a = a[idx][0].split('/')
        #     print(new_a)
            
        #     image_name = new_a[1][len(new_a[1].split('.')[0])+1:]
        #     print(image_name)
        #     with open(directory+new_a[0]+'.txt', 'r' ) as f:
        #         lines = f.readlines()
        #         #print(lines[1])
        #         for line in lines:
        #             if image_name in line:
        #                 print(line)
                        
        #                 col4.markdown(f" {line}")


                        
            #             col4.markdown(f" {line}")       


        # # add roi to known database
        #     if st.checkbox('add it to knonwn faces'):
        #         face_name = st.text_input('Name:', '')
                
        #         if st.button('add'):
        #             encoding = face_to_compare.tolist()
        #             DB.loc[len(DB)] = [face_name] + encoding
        #             DB.to_csv(PATH_DATA, index=False)
    else:
        st.write('No human face detected.')

    

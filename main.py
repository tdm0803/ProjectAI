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
from numpy import asarray
from retinaface import RetinaFace
from arcface import ArcFace
from os import listdir
import os
import torch.nn as nn
import torch

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

def extract_image(image):
    img1 = Image.open(image)            #open the image
    img1 = img1.convert('RGB')          #convert the image to RGB format
    return img1




def face_distance_to_conf(face_distance, face_match_threshold=0.4):
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * np.power((linear_val - 0.5) * 2, 0.2))



if __name__ == "__main__":

    directory = '/mnt/hdd2T/tham/projectAI/GUI/ProjectAI_Streamlit/crawl/downloads/'
    # image_byte = extract_image('/mnt/hdd2T/tham/projectAI/GUI/ProjectAI_Streamlit/data/downloads/Andrew Ng/1.andrew-ng-147a3890.jpg')
    for filename in listdir(directory):
        if '.txt'in filename:
            pass
        else:
            for image_name in listdir(directory+filename):
                
                path = directory + filename + '/'+ image_name
                
                name = filename + '/'+ image_name
                image_byte = extract_image(path)
                image_array = asarray(image_byte)   
                detector = RetinaFace()                 #assign the MTCNN detector
                f = detector.predict(image_array)
                max_faces =0
                rois = []
                try:
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



            # detect faces in the loaded image
    
                        
                    max_faces = len(f)
                    
                    

                    if max_faces > 0:
                        # select interested face in picture
                        for face_idx in range(max_faces):
                            #print(face_idx)
                            # try:
                            roi = rois[face_idx]
                            # st.image(BGR_to_RGB(roi), width=min(roi.shape[0], 400))

                            # initial database for known faces
                            DB = init_data()
                            face_encodings = DB[COLS_ENCODE].values
                            #dataframe = DB[COLS_INFO]
                            face_rec = ArcFace.ArcFace()
                            #print(face_rec)
                            face_to_compare = face_rec.calc_emb(roi)
                            # ln1 = nn.Linear(512,128)
                            # face_to_compare_128=ln1(torch.tensor(face_to_compare))
                            # new_encoding=face_to_compare_128.tolist()
                            
                            DB.loc[len(DB)] = [name] + face_to_compare.tolist()
                            DB.to_csv(PATH_DATA, index=False)
                except:
                    pass
                        


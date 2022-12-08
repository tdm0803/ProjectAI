# Project-AI-2022
With the rapid advance of web technologies, personal information leakage is extremely high ability. To detect personal information by identifying the degree of matching between faces and obtaining personal URL links, we first implement a web face and personal URL crawler and utilize the face recognition model. Face detection helps to locate the face on the frame and perform embedding vector extraction for recognition. This model combines the state-of-the-art RetinaFace for face detection and ArcFace for face recognition. Furthermore, we create a deletion system from "Remove an image from Google" to request removing any bad image following the related URL. 
Report an image link:https://support.google.com/websearch/answer/4628134?hl=en&fbclid=IwAR22qSSoOTW-DbWQ-QrMpaeRVvrtzpqma0X_Da_OZz27bJ_K15SxBVyPGj0

1) Create a database by crawling images and URLs from Google.
   python crawl.py 
2) Train our dataset with RetinaFace and ArcFace and save embedding vectors to csv file.
   python main.py
3) Test with input image pass over our network and run streamlit to show the results in GUI:
   streamlit run face_rec_main.py
4) Request remove the bad image based on Remove an image in Google by adding the bad image's link here: 
   https://support.google.com/websearch/answer/4628134?hl=en&fbclid=IwAR22qSSoOTW-DbWQ-QrMpaeRVvrtzpqma0X_Da_OZz27bJ_K15SxBVyPGj0
   

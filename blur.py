# # Importing Image class from PIL module
# from PIL import Image, ImageFilter
 
# # Opens a image in RGB mode
# im = Image.open(r"/mnt/hdd2T/tham/projectAI/GUI/ProjectAI_Streamlit/crawl/blur_dark_images/35_gates.webp")
 
# # Blurring the image
# im1 = im.filter(ImageFilter.BoxBlur(2))
 
# # Shows the image in image viewer
# im1 = im1.save("/mnt/hdd2T/tham/projectAI/GUI/ProjectAI_Streamlit/crawl/blur_dark_images/35_gates_blur.webp")


import urllib.request

with urllib.request.urlopen("https://www.usmagazine.com/wp-content/uploads/2020/05/Adeles-Amazing-Transformation-landing.jpg?quality=40&strip=all") as url:
    s = url.read()
    if 'malicious words'.encode('utf-8') in s: 
        print(s)
    # I'm guessing this would output the html source code ?
    # print(s)
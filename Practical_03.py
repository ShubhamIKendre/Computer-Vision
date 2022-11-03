import streamlit as st
from PIL import Image
import cv2
import numpy as np

#greyscale filter
def greyscale(img):
    greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return greyscale

# brightness adjustment
def bright(img, beta_value ):
    img_bright = cv2.convertScaleAbs(img, beta=beta_value)
    return img_bright

#sharp effect
def sharpen(img):
    kernel = np.array([[-1, -1, -1], [-1, 9.5, -1], [-1, -1, -1]])
    img_sharpen = cv2.filter2D(img, -1, kernel)
    return img_sharpen


#sepia effect
def sepia(img):
    img_sepia = np.array(img, dtype=np.float64) # converting to float to prevent loss
    img_sepia = cv2.transform(img_sepia, np.matrix([[0.272, 0.534, 0.131],
                                    [0.349, 0.686, 0.168],
                                    [0.393, 0.769, 0.189]])) # multipying image with special sepia matrix
    img_sepia[np.where(img_sepia > 255)] = 255 # normalizing values greater than 255 to 255
    img_sepia = np.array(img_sepia, dtype=np.uint8)
    return img_sepia


#grey pencil sketch effect
def pencil_sketch_grey(img):
    #inbuilt function to create sketch effect in colour and greyscale
    sk_gray, sk_color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.1)
    return  sk_gray

#colour pencil sketch effect
def pencil_sketch_col(img):
    #inbuilt function to create sketch effect in colour and greyscale
    sk_gray, sk_color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.1)
    return  sk_color


#HDR effect
def HDR(img):
    hdr = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
    return  hdr

# invert filter
def invert(img):
    inv = cv2.bitwise_not(img)
    return inv

#defining a function
from scipy.interpolate import UnivariateSpline
def LookupTable(x, y):
  spline = UnivariateSpline(x, y)
  return spline(range(256))

#summer effect
def Summer(img):
    increaseLookupTable = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    blue_channel, green_channel,red_channel  = cv2.split(img)
    red_channel = cv2.LUT(red_channel, increaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, decreaseLookupTable).astype(np.uint8)
    sum= cv2.merge((blue_channel, green_channel, red_channel ))
    return sum

#winter effect
def Winter(img):
    increaseLookupTable = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    blue_channel, green_channel,red_channel = cv2.split(img)
    red_channel = cv2.LUT(red_channel, decreaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, increaseLookupTable).astype(np.uint8)
    win= cv2.merge((blue_channel, green_channel, red_channel))
    return win

def Insta_Filter():
    st.subheader("Part A - Create your own Instagram Filter")

    image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])
    if not image_file:
        return None

    original_image = Image.open(image_file)
    original_image = np.array(original_image)

    st.image(original_image, caption="★ Original Image ★")

    st.text("______________________________________________________________________________________________")

    filter = st.radio(
        "➳ Choose your Favourite Filter 👇",
        ["GreySacle", "Sharpen", "Sepia", "Pencil_Sketch_Grey", "Pencil_Sketch_Col", "HDR", "Invert", "Summer",
         "Winter"],
        key="filter"
    )

    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>',
             unsafe_allow_html=True)

    st.text("______________________________________________________________________________________________")
    brightness_amount = st.slider("Brightness", min_value=-50, max_value=50, value=0)
    processed_image = bright(original_image, brightness_amount)

    if filter == "GreySacle":
        processed_image = greyscale(processed_image)
    elif filter == "Sharpen":
        processed_image = sharpen(processed_image)
    elif filter == "Sepia":
        processed_image = sepia(processed_image)
    elif filter == "Pencil_Sketch_Grey":
        processed_image = pencil_sketch_grey(processed_image)
    elif filter == "Pencil_Sketch_Col":
        processed_image = pencil_sketch_col(processed_image)
    elif filter == "HDR":
        processed_image = HDR(processed_image)
    elif filter == "Invert":
        processed_image == invert(processed_image)
    elif filter == "Summer":
        processed_image = Summer(processed_image)
    elif filter == "Winter":
        processed_image = Winter(processed_image)
    else:
        st.text("Sorry Filter is not Available")

    label = "✵ Result of %s Filter ✵" % filter
    st.image(processed_image, caption=label)


def main_opration():
    st.title("Compuetr Vision")
    st.header("Practical No. 03")

    Insta_Filter()
    st.text("_____________________________________________________________________________________________________________")

if __name__ == "__main__":
    main_opration()







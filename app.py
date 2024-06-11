# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn # ASGI server to run the FastAPI application.
from io import BytesIO # A class for working with binary data in memory.
from PIL import Image # A library for image processing.
from typing import Tuple # A library for type hints.
from fastapi.responses import HTMLResponse

import os
import numpy as np
import argparse
import warnings
import time
import cv2

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')

test_times = 1

MODEL_DIR = 'resources/anti_spoof_models'
CLASSES = ['RealFace', 'FakeFace']

def check_image(image):
    height, width, channel = image.shape
    print(image.shape)
    if width/height != 3/4:
        print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
        return False
    else:
        return True


def test(input_image, device_id):
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    image = input_image
    # result = check_image(image)
    # if result is False:
    #     return
    image_bbox = model_test.get_bbox(image)
    prediction = np.zeros((1, 3))
    test_speed = 0
    # sum the prediction from single model's result
    for model_name in os.listdir(MODEL_DIR):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        start = time.time()
        prediction += model_test.predict(img, os.path.join(MODEL_DIR, model_name))
        test_speed += time.time()-start

    # draw result of prediction
    label = np.argmax(prediction)
    value = prediction[0][label]/2

    # color = (255, 0, 0)
    # # Draw the rectangle on the image
    # cv2.rectangle(
    #     image,
    #     (image_bbox[0], image_bbox[1]),
    #     (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
    #     color,
    #     2
    # )

    # # Add the text label above the rectangle
    # cv2.putText(
    #     image,
    #     CLASSES[0] if label == 1 else CLASSES[1],
    #     (image_bbox[0], image_bbox[1] - 5),
    #     cv2.FONT_HERSHEY_COMPLEX,
    #     0.5 * image.shape[0] / 1024,
    #     color
    # )

    # # Convert the array back to an image and display it (for verification)
    # img_with_annotations = Image.fromarray(image)
    # img_with_annotations.save(f'images/result/image_{test_times}')
    return label, np.round(value, 4), image_bbox


def read_file_as_image(data) -> Tuple[np.ndarray, Tuple[int, int]]: # A function to read the image file as a numpy array
    # Load and convert the image
    img = Image.open(BytesIO(data)).convert('RGB')

    # Convert to NumPy array
    img_array = np.array(img)
    img_array = np.transpose(img_array, (1, 0, 2))
    img_array = img_array.astype(np.uint8)
    print("NumPy array shape:", img_array.shape)  # Should be (height, width, channels)
    print("NumPy array dtype:", img_array.dtype)  # Should be uint8
    return img_array # Return the image

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict") # A decorator to create a route for the predict endpoint
async def predict(file: UploadFile = File(...)): # The function that will be executed when the endpoint is called
    try: # A try block to handle any errors that may occur
        image = read_file_as_image(await file.read()) # Read the image file

        predicted_class, confidence, box_index = test(image, 0) # Make a prediction
 
        return { # Return the prediction
            'class': CLASSES[0] if predicted_class == 1 else CLASSES[1],   
            'confidence': float(confidence),
            'box_index': box_index
        }
    except Exception as e: # If an error occurs
        raise HTTPException(status_code=400, detail=str(e)) # Raise an HTTPException with the error message
    

if __name__ == "__main__": # If the script is run directly
    uvicorn.run(app, host="localhost", port=8002) # Run the FastAPI app using uvicorn
import cv2
import numpy as np

def pose_estimation(input_image):
    '''
    Expected Input for model: 1x3x256x456, BGR

    1. Resize image. OpenCV resize function take in format (w,h)

    2. Since OpenCV load image with format  height, width, channels = img.shape (HWC)
       Swap from HWC to CHW: preprocessed_image.transpose((2,0,1))

    3. Add batch dimension of "1"  with preprocessed_image = preprocessed_image.reshape(1,3,256,456)

    '''

    # resize image: This is on the form 
    preprocessed_image = cv2.resize( input_image, (456,256)) # width, height 
    
    #Swap from HWC to CHW
    preprocessed_image.transpose((2,0,1))
    # Numpy:
    # preprocessed_image = np.rollaxis(preprocessed_image,2,0)
    
    # Add batch dimension
    preprocessed_image = preprocessed_image.reshape(1,3,256,456)
    #Numpy:
    #preprocessed_image = preprocessed_image[np.newaxis, ...]
    return preprocessed_image


def text_detection(input_image):
    '''
    Expected Input for model: 1x3x768x1280, BGR

    1. Resize image. OpenCV resize function take in format (w,h)

    2. Since OpenCV load image with format  height, width, channels = img.shape (HWC)
       Swap from HWC to CHW: preprocessed_image.transpose((2,0,1))

    3. Add batch dimension of "1"  with preprocessed_image = preprocessed_image.reshape(1,3,768,1280)
    '''

       # resize image: This is on the form 
    preprocessed_image = cv2.resize( input_image, (768,1280)) # width, height 
    
    #Swap from HWC to CHW
    preprocessed_image.transpose((2,0,1))
    # Numpy:
    # preprocessed_image = np.rollaxis(preprocessed_image,2,0)
    
    # Add batch dimension
    preprocessed_image = preprocessed_image.reshape(1,3,768,1280)
    #Numpy:
    #preprocessed_image = preprocessed_image[np.newaxis, ...]
    
    return preprocessed_image


def car_meta(input_image):
        '''
    Expected Input for model: 1x3x72x72, BGR

    1. Resize image. OpenCV resize function take in format (w,h)

    2. Since OpenCV load image with format  height, width, channels = img.shape (HWC)
       Swap from HWC to CHW: preprocessed_image.transpose((2,0,1))

    3. Add batch dimension of "1"  with preprocessed_image = preprocessed_image.reshape(1,3,768,1280)
    '''

       # resize image: This is on the form 
    preprocessed_image = cv2.resize( input_image, (72,72)) # width, height 
    
    #Swap from HWC to CHW
    preprocessed_image.transpose((2,0,1))
    # Numpy:
    # preprocessed_image = np.rollaxis(preprocessed_image,2,0)
    
    # Add batch dimension
    preprocessed_image = preprocessed_image.reshape(1,3,72,72)
    #Numpy:
    #preprocessed_image = preprocessed_image[np.newaxis, ...]
    
    return preprocessed_image
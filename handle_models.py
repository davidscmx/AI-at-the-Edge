import cv2
import numpy as np


def handle_pose(output, input_shape):
    '''
    Handles the output of the Pose Estimation model.
    Returns ONLY the keypoint heatmaps, and not the Part Affinity Fields.
    '''
    # Extract only the second blob output (keypoint heatmaps)
    keypoint_heatmaps = output["Mconv7_stage2_L2"]
    # Resize the heatmap back to the size of the input
    
    input_width = input_shape[1]
    input_height = input_shape[0]
    
    keypoint_heatmaps_resized = np.zeros((1,19,input_height, input_width))
    for i in range(19):
        tmp_keypoint = output["Mconv7_stage2_L2"][0, i, :, :] 
        tmp_keypoint = cv2.resize(tmp_keypoint, (input_width, input_height))
        keypoint_heatmaps_resized[0,i,...] = tmp_keypoint
        
    keypoint_heatmaps_resized = np.moveaxis(keypoint_heatmaps_resized, 0,3)
    # return [1,19,750,1000]
    return keypoint_heatmaps_resized


def handle_text(output, input_shape):
    '''
    Handles the output of the Text Detection model.
    Returns ONLY the text/no text classification of each pixel,
        and not the linkage between pixels and their neighbors.
    
    https://docs.openvinotoolkit.org/2019_R3/_models_intel_text_detection_0004_description_text_detection_0004.html
    
    [1x2x192x320] - output: logits related to text/no-text classification for each pixel.
    [1x16x192x320] - output: related to linkage between pixels and their neighbors.
    '''
    input_height = input_shape[0]
    input_width  = input_shape[1]

    # Extract only the first blob output (text/no text classification)
    sem_output_text = output["model/segm_logits/add"][0, 0, :, :] 
    sem_output_notext = output["model/segm_logits/add"][0, 1, :, :]
    
    # Resize this output back to the size of the input
    sem_output_text = cv2.resize(sem_output_text, (input_width, input_height))
    sem_output_notext = cv2.resize(sem_output_notext,  (input_width, input_height))

    return [sem_output_notext,sem_output_text]


def handle_car(output, input_shape):
    '''
    Handles the output of the Car Metadata model.
    Returns two integers: the argmax of each softmax output.
    The first is for color, and the second for type.
    '''
    color_max_index = np.argmax(output["color"])
    type_max_index = np.argmax(output["type"])
    
    return [color_max_index,type_max_index]


def handle_output(model_type):
    '''
    Returns the related function to handle an output,
        based on the model_type being used.
    '''
    if model_type == "POSE":
        return handle_pose
    elif model_type == "TEXT":
        return handle_text
    elif model_type == "CAR_META":
        return handle_car
    else:
        return None


'''
The below function is carried over from the previous exercise.
You just need to call it appropriately in `app.py` to preprocess
the input image.
'''
def preprocessing(input_image, height, width):
    '''
    Given an input image, height and width:
    - Resize to width and height
    - Transpose the final "channel" dimension to be first
    - Reshape the image to add a "batch" of 1 at the start 
    '''
    image = np.copy(input_image)
    image = cv2.resize(image, (width, height))
    image = image.transpose((2,0,1))
    image = image.reshape(1, 3, height, width)

    return image
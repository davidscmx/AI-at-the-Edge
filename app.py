import argparse
import cv2
import numpy as np

from handle_models import handle_output, preprocessing
from inference import Network

CAR_COLORS = ["white", "gray", "yellow", "red", "green", "blue", "black"]
CAR_TYPES = ["car", "bus", "truck", "van"]


def get_args():
    '''
    Gets the arguments from the command line.
    '''
    
    parser = argparse.ArgumentParser("Basic Edge App with Inference Engine")
    # -- Create the descriptions for the commands

    cpu_extension_description = "CPU extension file location, if applicable"
    device_description = "Device, if not CPU (GPU, FPGA, MYRIAD)"
    input_description = "The location of the input image"
    model_xml_description = "The location of the model XML file"
    type_of_model_description = "The type of model: POSE, TEXT or CAR_META"

    conf_thresh_description = "Confidence threshold used for the bounding boxes" 
    conf_thresh_description = "Confidence threshold used for the bounding boxes" 
    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    required.add_argument("-input", help=input_description, required=True)
    required.add_argument("-model", help=model_xml_description, required=True)
    required.add_argument("-type", help=type_of_model_description, required=True)
    optional.add_argument("-cpu", help=cpu_extension_description, default=None)
    optional.add_argument("-device", help=device_description, default="CPU")
    args = parser.parse_args()

    return args


def get_mask(processed_output):
    '''
    Given an input image size and processed output for a semantic mask,
    returns a masks able to be combined with the original image.
    '''
    # Create an empty array for other color channels of mask
    empty = np.zeros(processed_output.shape)
    # Stack to make a Green mask where text detected
    mask = np.dstack((empty, processed_output, empty))

    return mask


def draw_boxes(frame, result, args, width, height):
    '''
    Draw bounding boxes onto the frame.
    '''
    for box in result[0][0]: # Output shape is 1x1x100x7
        conf = box[2]
        if conf >= 0.5:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
    return frame

def create_output_image(model_type, image, output):
    '''
    Using the model type, input image, and processed output,
    creates an output image showing the result of inference.
    '''
    if model_type == "POSE":
        # Remove final part of output not used for heatmaps
        output = output[:-1]
        # Get only pose detections above 0.5 confidence, set to 255
        for c in range(len(output)):
            output[c] = np.where(output[c]>0.5, 255, 0)
        # Sum along the "class" axis
        output = np.sum(output, axis=0)
        # Get semantic mask
        pose_mask = get_mask(output)
        # Combine with original image
        image = image + pose_mask
        return image
    elif model_type == "TEXT":
        # Get only text detections above 0.5 confidence, set to 255
        output = np.where(output[1]>0.5, 255, 0)
        # Get semantic mask
        text_mask = get_mask(output)
        # Add the mask to the image
        image = image + text_mask
        return image
    elif model_type == "CAR_META":
        # Get the color and car type from their lists
        color = CAR_COLORS[output[0]]
        car_type = CAR_TYPES[output[1]]
        print(color, car_type)
        # Scale the output text by the image shape
        scaler = max(int(image.shape[0] / 1000), 1)
        # Write the text of color and type onto the image
        image = cv2.putText(image, 
            "Color: {}, Type: {}".format(color, car_type), 
            (50 * scaler, 100 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 
            2 * scaler, (255, 255, 255), 3 * scaler)
        return image
    else:
        print("Unknown model type, unable to create output image.")
        return image


def perform_inference(args):
    '''
    Performs inference on an input image/input stream, given a model.
    '''
    image_flag = False
    CODEC = 0x00000021
    # Create a Network for using the Inference Engine
    inference_network = Network()
    # Load the model in the network, and obtain its input shape
    inference_network.load_model(args.model, args.device, args.cpu)
    net_input_shape = inference_network.get_input_shape() 
    
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)

    if cap.isOpened():
        image_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        image_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not image_flag:
        out = cv2.VideoWriter('out.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 
                              30, (image_width, image_height)
    else:
        out = None

    counter_img = 0
    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
       
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        ### Preprocess the input image
        preprocessed_frame = preprocessing(frame, h, w)

        # Perform synchronous inference on the image
        if do_async:
            inference_network.async_inference(preprocessed_frame)
            ### Get the output of inference
            if inference_network.wait() == 0:
            result = inference_network.extract_output()
            ### Update the frame to include detected bounding boxes
            frame = draw_boxes(frame, result, args, image_width, image_height)
            # Write out the frame
            out.write(frame)

        else:
            inference_network.sync_inference(preprocessed_image)
            # Obtain the output of the inference request
            output = inference_network.extract_output()

        ### Handle the output of the network, based on args.type
        ### Note: This will require using `handle_output` to get the correct
        ### function, and then feeding the output to that function.
        processed_output = handle_output(args.type)(output, image.shape)

        output_image = create_output_image(args.type, image, processed_output)

        #cv2.namedWindow("frame",cv2.WINDOW_NORMAL)
        #cv2.imshow('frame', output_image)
        #cv2.waitKey(15)
        
        cv2.imwrite("outputs/{}-output.png".format(counter_img), output_image)
        counter_img +=1
        #if image_flag:
        #    cv2.imwrite("outputs/{}-output.png".format(args.type), output_image)
        #else:
        #    out.write(output_image)
	
    out.release()
    cap.release()
    #Closes all the frames
    cv2.destroyAllWindows() 

def main():
    args = get_args()
    perform_inference(args)

if __name__ == "__main__":
    main()

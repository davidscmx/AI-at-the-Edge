import argparse
import sys
import cv2
import numpy as np
from random import randint

from handle_models import handle_output, preprocessing
from inference import Network

import paho.mqtt.client as mqtt
import ffmpeg
import time
from inspect import currentframe, getframeinfo
import logging 
from utilities.draw_utilities import *
import socket
import json

HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

CPU_EXTENSION = "/home/david/Programs/openvino_2019/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Basic Edge App with Inference Engine")

    cpu_extension_description = "CPU extension file location, if applicable"
    device_description = "Device, if not CPU (GPU, FPGA, MYRIAD)"
    input_description = "The location of the input image"
    model_xml_description = "The location of the model XML file"
    type_of_model_description = "The type of model: POSE, TEXT, CAR_META, ADAS"

    conf_thresh_description = "Confidence threshold used for the bounding boxes" 
    conf_thresh_description = "Confidence threshold used for the bounding boxes" 

    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    required.add_argument("-input", help=input_description, required=True)
    required.add_argument("-model", help=model_xml_description, required=True)
    required.add_argument("-type", help=type_of_model_description, required=True)

    optional.add_argument("-device", help=device_description, default="CPU")
    args = parser.parse_args()

    return args

def perform_inference(args):
    '''
    Performs inference on an input image/input stream, given a model.
    '''
    mqtt_client = mqtt.Client()
    mqtt_client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    image_flag = False
    CODEC = 0x00000021

    inference_network = Network()
    inference_network.load_model(args.model, args.device, CPU_EXTENSION)
    net_input_shape = inference_network.get_input_shape() 
    net_input_height = net_input_shape[2]
    net_input_width =  net_input_shape[3]

    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)

    if cap.isOpened():
        image_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        image_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not image_flag:
        out = cv2.VideoWriter('out.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (image_width, image_height))
    else:
        out = None

    counter_img = 0
    while cap.isOpened():
        flag, frame = cap.read()
       
        if not flag:
            break
        key_pressed = cv2.waitKey(1000)

        preprocessed_frame = preprocessing(frame, net_input_height, net_input_width)
        do_async = True
        if do_async:
            inference_network.async_inference(preprocessed_frame)
            if inference_network.wait() == 0:
                result = inference_network.extract_output()
                output_frame, classes = draw_masks(result, image_width, image_height)
                class_names = get_class_names(classes)
                speed = randint(50,70)
            
                mqtt_client.publish("class", json.dumps({"class_names": class_names}))
                mqtt_client.publish("speedometer", json.dumps({"speed": speed}))
        
        #print(output_frame.shape)
        #show_frame(output_frame)
        sys.stdout.buffer.write(output_frame)
        sys.stdout.flush()
  
    out.release()
    cap.release()
    cv2.destroyAllWindows() 

    mqtt_client.disconnect()

def main():
    args = get_args()
    perform_inference(args)

if __name__ == "__main__":
    main()

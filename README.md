# AI at the Edge 



## Pre-requisites
The course uses the 2019 R3 version of the OpenVino toolkit, 
There may be some variance in syntax from newer versions.

Install pre-requisites
sudo ./install_openvino_dependencies.sh


### 


./bin/setupvars.sh

## What a mess with the environemt

python 3.6
numpy 1.19.1
OpenCV from OpenVino


### 2019.3 Version 

Release Notes: https://software.intel.com/content/www/us/en/develop/articles/openvino-relnotes-2019.html

Correct Intel website: 
https://download.01.org/opencv/2019/openvinotoolkit/R3/l_openvino_toolkit_runtime_ubuntu18_p_2019.3.376.tgz

Note: The register then download webpage is not working. 

### Latest Version

https://software.intel.com/content/www/us/en/develop/articles/openvino-relnotes.html

sudo apt update


### Downloading Pre-trained Model

./deployment_tools/tools/model_downloader/downloader.py


### Converting a TensorFlow model to an Intermediate Representation

#### --reverse_input_channels RGB->BGR

```
python ./openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
```

### Converting a Caffe model to an Intermediate Representation
```
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model squeezenet_v1.1.caffemodel --input_proto deploy.prototxt
```

```
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model model.onnx
```

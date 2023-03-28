

# import the opencv library
import cv2
import matplotlib.pyplot as plt
import numpy as np
from openvino.runtime import Core
import operator



def get_crop_image(image, box):
    xmin, ymin, xmax, ymax = box
    crop_image = image[ymin:ymax, xmin:xmax]
    return crop_image



# lendo e compilando o modelo
ie = Core()

model_face = ie.read_model(model="model/face-detection-retail-0004.xml")
compiled_face_model = ie.compile_model(model=model_face, device_name="CPU")
input_face_layer = compiled_face_model.input(0)
output_face_layers = compiled_face_model.output(0)

model_emotions = ie.read_model(model="model/emotions-recognition-retail-0003.xml")
compiled_emotions_model = ie.compile_model(model=model_emotions, device_name="CPU")

input_layer_emotions_ir = compiled_emotions_model.input(0)
output_layer_emotions_ir = compiled_emotions_model.output()




def openvino_emotion(frame):
    N, C, H, W = input_face_layer.shape

    # Resize the image to meet network expected input sizes.
    resized_image = cv2.resize(frame, (W, H))

    input_image = np.expand_dims(resized_image.transpose(2, 0, 1), 0)

    result = compiled_face_model([input_image])[output_face_layers]

    x_min = int(result[0,0,0,3]*frame.shape[1])
    y_min = int(result[0,0,0,4]*frame.shape[0])

    x_max = int(result[0,0,0,5]*frame.shape[1])
    y_max = int(result[0,0,0,6]*frame.shape[0])

    

    box = x_min,y_min,x_max,y_max
    face = get_crop_image(frame,box)

    N, C, H, W = input_layer_emotions_ir.shape
    resized_face_image = cv2.resize(face, (W, H))
    input_face_image = np.expand_dims(resized_face_image.transpose(2, 0, 1), 0)
    values = compiled_emotions_model([input_face_image])[output_layer_emotions_ir]
    emotions =  ['neutral', 'happy', 'sad', 'surprise', 'anger']
    res = dict(zip(emotions, list(values[0])))
    emotion = max(res.items(), key=operator.itemgetter(1))[0]


    cv2.rectangle(frame, (x_min,y_min), (x_max,y_max), (0,0,255), 2)
    cv2.putText(frame, f'{emotion}',(x_max,y_max-10),cv2.FONT_HERSHEY_PLAIN,3,(0,0,0),2)


    return frame


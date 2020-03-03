import numpy as np 
import cv2 
import time, os, sys
from tqdm import tqdm
from skimage import io
import matplotlib.pyplot as plt 
from PIL import Image, ImageDraw
import scipy.io

from dataset.dataset import *

#Configuration settings 
FPS = 4
wait_time_seconds = 1./FPS

cv2.namedWindow("DLP_3000",cv2.WND_PROP_FULLSCREEN)
cv2.moveWindow("DLP_3000",1290 + 912 + 1,30)
cv2.setWindowProperty("DLP_3000",cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cv2.namedWindow("DLP_4500",cv2.WND_PROP_FULLSCREEN)
cv2.moveWindow("DLP_4500",1300,10)
cv2.setWindowProperty("DLP_4500",cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

#Create arduino trigger device
from trigger import trigger 
Trigger = trigger()

offset = 2
dmd_3000_x = 684-offset
dmd_3000_y = 608-offset
dmd_4500_x = 1140-offset
dmd_4500_y = 912-offset
array_3000 = np.ones((dmd_3000_x,dmd_3000_y)) # all off
array_4500 = np.ones((dmd_4500_x,dmd_4500_y)) # all off

#Here we have the limits of the dmd as desired by matlab
red_start = 149
red_end = 249 # 339 # inclusive
blue_start = 287 # 247 
blue_end = 397 # inclusive


which_row_of_dmd_3000 = 342
which_column_of_dmd_4500 = 460

first_float, y_test = mnist_inputs()
second_float = mnist_weights()

num_images = first_float.shape[0]
software_output_activation = second_float.shape[1]
input_activation = first_float.shape[1]
num_bits = 8

get_background = False
get_allon = False
send_matricies = True

# number of frames to collect for background subtraction
num_frames_background_average = 100
num_frames_allon_average = 100

#Take floating point input matrix and quantize it to 8-bit integers
def quantize_inputs(input_mat):
    #Our quantization scheme maps from floating point space (fmin - fmax) to quantization space (0 - 255)
    #It does so with a linear mapping using a scale and zero point as described below
    fmax = input_mat.max()
    fmin = input_mat.min()
    qmax = 2**(num_bits) - 1
    qmin = 0
    input_scale = (fmax - fmin) / (qmax - qmin)
    input_z = qmin - fmin/input_scale 

    q_input = input_z + input_mat / input_scale 
    q_input = q_input.astype(np.uint8)

    #We now have an integer array of size (batch, input_activation) 
    #We wish to convert this to binary array of size (batch, input_activation, num_bits)
    q_input = np.expand_dims(q_input,axis=2) #Add an extra dimension
    q_input = np.unpackbits(q_input,axis=2) #Expand integer to binary in that dimension

    return q_input, input_scale, input_z

def quantize_weights(weight_mat):
    fmax = weight_mat.max()
    fmin = weight_mat.min()
    qmax = 2**(num_bits) - 1
    qmin = 0
    weight_scale = (fmax - fmin) / (qmax - qmin)
    weight_z = qmin - fmin/weight_scale 

    q_weight = weight_z + weight_mat / weight_scale 
    q_weight = q_weight.astype(np.uint8)

    q_weight = np.expand_dims(q_weight,axis=2)
    q_weight = np.unpackbits(q_weight,axis=2)

    return q_weight, weight_scale, weight_z 

#Quantize inputs and weights
input_mat,input_scale,input_z = quantize_inputs(first_float)
weight_mat,weight_scale,weight_z = quantize_weights(second_float)

print(input_scale, input_z)
print(weight_scale,weight_z)

#For layer one
# scipy.io.savemat("mnist_three_layer_l1.mat",{"first_float":first_float,
# "second_float":second_float,"y_test":y_test,"input_scale":input_scale,
# "input_z":input_z,"weight_scale":weight_scale,"weight_z":weight_z,
# "input_mat":input_mat.astype(np.float),"weight_mat":weight_mat.astype(np.float)})

#For layer two
scipy.io.savemat("mnist_three_layer_l2.mat",{"second_layer_input":first_float,
"second_layer_weight":second_float,"y_test":y_test,"input_scale":input_scale,
"input_z":input_z,"weight_scale":weight_scale,"weight_z":weight_z,
"second_layer_input_quantized":input_mat.astype(np.float),"second_layer_weight_quantized":weight_mat.astype(np.float)})


#Get background image
if get_background:
    print("Getting current background image")
    for t in tqdm(range(num_frames_background_average)):
        display_time = time.time()

        array_3000 = np.ones((dmd_3000_x,dmd_3000_y)) # all off
        array_4500 = np.ones((dmd_4500_x,dmd_4500_y)) # all off

        array_4500 = np.flipud(array_4500) # displays inverted image, so have to flip
        
        cv2.imshow("DLP_3000", array_3000)
        cv2.imshow("DLP_4500", array_4500)
        key = cv2.waitKey(15)

        while time.time() - display_time < wait_time_seconds:
            time.sleep(0.001)

        Trigger.send_trigger_pulse()
    if get_allon or send_matricies:
        input("Did you get background?")

if get_allon:
    #Get allon image
    print("Getting current all on image")
    for t in tqdm(range(num_frames_allon_average)):
        display_time = time.time()
        # DMD3000.format_data(np.ones(hardware_batch))
        # DMD4500.format_data(np.ones(hardware_output_activation))

        array_3000 = np.ones((dmd_3000_x,dmd_3000_y)) # all off
        array_4500 = np.ones((dmd_4500_x,dmd_4500_y)) # all off

        array_3000[which_row_of_dmd_3000,red_start:red_end+1:1] = np.zeros((1, red_end-red_start+1))
        array_4500[blue_start*2:(blue_end+1)*2:2,which_column_of_dmd_4500] = np.zeros((blue_end-blue_start+1))

        array_4500 = np.flipud(array_4500) # displays inverted image, so have to flip
        
        cv2.imshow("DLP_3000", array_3000)
        cv2.imshow("DLP_4500", array_4500)
        key = cv2.waitKey(15)

        while time.time() - display_time < wait_time_seconds:
            time.sleep(0.001)

        Trigger.send_trigger_pulse()
    if send_matricies:
        input("Did you get allon?")

#Send pseudorandom data
number_timesteps = input_activation * num_bits

if send_matricies:
    print("Sending matrix data")
    for t in tqdm(range(number_timesteps)):
        act_num = t//num_bits
        bit_num = t%num_bits
        display_time = time.time()

        array_3000 = np.ones((dmd_3000_x,dmd_3000_y)) # all off
        array_4500 = np.ones((dmd_4500_x,dmd_4500_y)) # all off

        #Here we place the appropriate matrix data into the vector
        #input_mat is (batch,input_act,num_bits)
        #weight_mat is (input_act,output_act,num_bits)
        to_display_3000 = np.ones((red_end-red_start+1))
        to_display_3000[:num_images] = 1- input_mat[:,act_num,bit_num]

        to_display_4500 = np.ones((blue_end-blue_start+1))
        to_display_4500[:software_output_activation] = 1-weight_mat[act_num,:,bit_num]

        array_3000[which_row_of_dmd_3000,red_start:red_end+1:1] = to_display_3000
        array_4500[blue_start*2:(blue_end+1)*2:2,which_column_of_dmd_4500] = to_display_4500

        array_4500 = np.flipud(array_4500) # displays inverted image, so have to flip
        
        cv2.imshow("DLP_3000", array_3000)
        cv2.imshow("DLP_4500", array_4500)
        key = cv2.waitKey(15)

        while time.time() - display_time < wait_time_seconds:
            time.sleep(0.001)

        Trigger.send_trigger_pulse()

    a = input("Here stop the recording")
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
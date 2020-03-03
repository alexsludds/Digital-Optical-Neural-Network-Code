#Here we are going to compress the size of MNIST such that the number of input neurons is reduced
import tensorflow as tf
import numpy as np
import scipy as sp
import scipy.io as io
import scipy.misc as misc
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)
desired_size = (7,7)

#interpolate images
def interp_mnist(input_tensor,desired_size):
    output_tensor = np.zeros((input_tensor.shape[0],desired_size[0]*desired_size[1]))
    for i in range(input_tensor.shape[0]):
        output_tensor[i,:] = misc.imresize(input_tensor[i,:,:],desired_size,interp='cubic').flatten()
    return output_tensor

small_train_images = interp_mnist(train_images,desired_size)
small_test_images = interp_mnist(test_images,desired_size)

def data():
    """
    Data providing function:

    This function is separated from create_model() so that hyperopt
    won't reload data for each evaluation run.
    """
    desired_size = (7,7)
    def interp_mnist(input_tensor,desired_size):
        output_tensor = np.zeros((input_tensor.shape[0],desired_size[0]*desired_size[1]))
        for i in range(input_tensor.shape[0]):
            output_tensor[i,:] = misc.imresize(input_tensor[i,:,:],desired_size,interp="bilinear").flatten()
        return output_tensor
    
    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = interp_mnist(x_train,desired_size)
    x_test = interp_mnist(x_test,desired_size)
    nb_classes = 10
    y_train = tf.keras.utils.to_categorical(y_train, nb_classes)
    y_test = tf.keras.utils.to_categorical(y_test, nb_classes)
    return x_train, y_train, x_test, y_test


def create_model(x_train, y_train, x_test, y_test):
    """
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100, input_shape=(desired_size[0]*desired_size[1],), use_bias=False))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Activation(activation=tf.nn.relu))
    
    model.add(tf.keras.layers.Dense(100, use_bias=False))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Activation(activation=tf.nn.relu))
    
    model.add(tf.keras.layers.Dense(10, use_bias=False))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Activation(activation=tf.nn.softmax))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                  optimizer='adam')

    result = model.fit(x_train, y_train,
              batch_size=100,
              epochs=12,
              validation_data=(x_test,y_test))
    
    #get the highest validation accuracy of the training epochs
    validation_acc = np.amax(result.history['val_acc']) 
    return model

x_train, y_train, x_test, y_test = data()
model = create_model(x_train, y_train, x_test, y_test)

np.save("storage/x_test",x_test[:900])
np.save("storage/y_test",y_test[:900])
first_weight = model.get_weights()[0]
second_weight = model.get_weights()[1]
final_weight = model.get_weights()[2]

np.save("storage/first_weight",first_weight)
np.save("storage/second_weight",second_weight)
np.save("storage/final_weight",final_weight)

#Here we save the weights as a matlab file as well.
io.savemat("storage/mnist_weights",{"second_weight":second_weight,'first_weight':first_weight,"final_weight":final_weight})
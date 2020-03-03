# Digital-Optical-Neural-Network-Code
Relevent code snippets for the digital optical neural network project in Dirk Englund's group

To use this code I have included version-stamped versions of the required code in the req.txt file.
To install these requirements easily please run:

python -m pip install -r req.txt
This command takes your version of python, references the version of pip tied to it, and recursively installs all packages inside of the txt file and their dependencies.

Relevent code on how to generate a model with the same hyperparameters as what we used in our experiment are found in generate_network_and_dataset.py
Code on the method we use for qunatization can be found in quantize.py
The code for running the hardware at the time of writing this document are in the running_experiment_code.py file.
If you want to use this code you also need the corresponding arduino code located in server.ino. The arduino is acting as a simple pulse generator. It takes in a serial command saying "generate a pulse" and makes a 5ms pulse. There is an amplifier (tiny op-amp circuit) to drive the trigger pin of the camera to the required voltage.

If you have any problems feel free to email me :    asludds (at) mit (dot) edu or my colleague:  lbern (at) mit (dot) edu
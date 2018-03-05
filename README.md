# Implementationn and investigation of Fully Connected and Convolutional neural networks on the CIFAR-10 dataset. #
* [Fully Connected Neural Network](https://github.com/alexvlis/cnn-cifar10/blob/master/FullyConnectedNets.ipynb)
	* Stochastic Gradient Descent
	* SGD with Momentum
	* RMSProp
	* Adam
* [Batch Normalization](https://github.com/alexvlis/cnn-cifar10/blob/master/BatchNormalization.ipynb)
* [Dropout](https://github.com/alexvlis/cnn-cifar10/blob/master/Dropout.ipynb)
* [Convolutional Neural Network](https://github.com/alexvlis/cnn-cifar10/blob/master/ConvolutionalNetworks.ipynb)


## Setup

**Install virtual environment:**
```bash
sudo pip install virtualenv      # This may already be installed
virtualenv .env                  # Create a virtual environment
source .env/bin/activate         # Activate the virtual environment
pip install -r requirements.txt  # Install dependencies

deactivate                       # Exit the virtual environment
```

**Download data:** Run the following to download the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html):
```bash
cd deeplearning/datasets
./get_datasets.sh
```

**Compile the Cython extension:** Convolutional Neural Networks require a very
efficient implementation. You will need to compile the Cython extension
before you can run the code. From the `deeplearning` directory, run the following
command:

```bash
python setup.py build_ext --inplace
```

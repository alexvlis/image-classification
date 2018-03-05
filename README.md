# Implementationn and investigation of Fully Connected and Convolutional neural networks on the CIFAR-10 dataset. $
* [Fully Connected Neural Network](https://github.com/alexvlis/cnn-cifar10/blob/master/FullyConnectedNets.ipynb)
	* Stochastic Gradient Descent
	* SGD with Momentum
	* RMSProp
	* Adam
* [Batch Normalization](https://github.com/alexvlis/cnn-cifar10/blob/master/BatchNormalization.ipynb)
* [Dropout](https://github.com/alexvlis/cnn-cifar10/blob/master/Dropout.ipynb)
* [Convolutional Neural Network](https://github.com/alexvlis/cnn-cifar10/blob/master/ConvolutionalNetworks.ipynb)


## Setup
Make sure your machine is set up with the assignment dependencies. 

**[Option 2] Manual install, virtual environment:**
If you do not want to use Anaconda and want to go with a more manual and risky
installation route you will likely want to create a
[virtual environment](http://docs.python-guide.org/en/latest/dev/virtualenvs/)
for the project. If you choose not to use a virtual environment, it is up to you
to make sure that all dependencies for the code are installed globally on your
machine. To set up a virtual environment, run the following:

```bash
cd assignment1
sudo pip install virtualenv      # This may already be installed
virtualenv .env                  # Create a virtual environment
source .env/bin/activate         # Activate the virtual environment
pip install -r requirements.txt  # Install dependencies
# Work on the assignment for a while ...
deactivate                       # Exit the virtual environment
```

**Download data:**
Once you have the starter code, you will need to download the CIFAR-10 dataset.
Run the following from the `assignment1` directory:

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

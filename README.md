# DroneSegmentation

Tools for training Caffe models for Semantic Segmentation of UAV imagery at PrendingerLab, NII.

Originally developed by Andrew Holliday, Johannes Laurmaa, Pierre Ecarlat, Kim
Samba and Chetak Kandaswamy.

The work of the Deep Drone team primarily consists of experimenting with convolutional neural network architectures.  We obtain existing architectures from others and develop new ones ourselves.  We train networks with these architectures on a variety of datasets, and then measure the accuracy and time-performance of the networks.  Primarily the networks we work with perform semantic-segmentation - that is, they take an image as input, and output a meaningful label for each pixel of the image (for instance, whether that pixel is part of a cat, a dog, a tree, etc.)

For our work with convolutional neural nets (CNNs), we use the [Caffe](http://caffe.berkeleyvision.org/) framework.  Caffe is unique in that one can define and train a network without having to do any actual programming.  Caffe network architectures and training parameters are all defined in .prototxt configuration files which are fed to the Caffe binary, which trains a network and outputs the weights as a .caffemodel file.  If you’re not familiar with Caffe, it is highly recommended that you work through the below two tutorials to gain some familiarity with the system:

[http://caffe.berkeleyvision.org/gathered/examples/cifar10.html](http://caffe.berkeleyvision.org/gathered/examples/cifar10.html)

[http://caffe.berkeleyvision.org/gathered/examples/finetune_flickr_style.html](http://caffe.berkeleyvision.org/gathered/examples/finetune_flickr_style.html)

To expand on the basic features of Caffe, we have borrowed, altered, and written from-scratch a considerable amount of Python code that makes use of Caffe’s Python bindings.

## Repository Structure

### caffeUtils

caffeUtils is a python module that provide a variety utilities for deep learning with Caffe.  Most of these are based on code provided in [the FCN github project](https://github.com/shelhamer/fcn.berkeleyvision.org), but are heavily modified by us.  Make sure that you add the location of this repo to your PYTHONPATH environment variable, as described in the section "Environment Setup".

### caffeTools

caffeTools is where we keep most of the scripts that we use to train, test, and evaluate Caffe networks, as well as to manipulate the datasets.  Most of the scripts here make use of the caffeUtils module.  Usage information for all of them can be found by running them with the -h option, and a more detailed description of their purpose and usage can be found in the folder's README.md file.

### transferLearningFramework

System for performing transfer-learning and multi-source training with Caffe.

* protobuf/transferLearning.proto: defines the config file format for describing a transfer-learning or multi-source learning job.

* scripts/transfer.py: the main script that takes a .prototxt config file (as defined in transferLearning.proto) as input, and carries it out.

* examples/: this directory contains various test cases and examples of configuration files for the transfer-learning framework.  Useful to study these to understand how to use this tool.

* configure: a simple bash script that auto-generates protocol-buffer python files from .proto message definition files.  When you first check out this repo, this script will need to be run before you can use some of the other scripts that rely on it.  You’ll also need to run it again if you make changes to any .proto files in this repo.

### models

Various Caffe configuration files that define the network architectures and solving procedures that we use to train our models, as well as multi-source jobs we use (in the multiSource/ subdirectory).

## Dependencies

Mainly *caffe* and its own dependencies but *CUDA* and *cuDNN* are essential. A [guide](https://github.com/BVLC/caffe/wiki/Ubuntu-16.04-or-15.10-Installation-Guide) for its installation on Ubuntu 16.04 is available.

 - [caffe](http://caffe.berkeleyvision.org/)
 - [CUDA](https://developer.nvidia.com/cuda-downloads)
 - [cuDNN](https://developer.nvidia.com/cudnn)

### Typical file structure around the project

* caffe - the code and compiled binaries of the Caffe deep learning framework.

* caffeSegNet - the SegNet project, along with its special modified version of Caffe. #TODO why needed?

* DroneSegmentation - this repository itself, cloned in the host computer whre the code will run.

* data - where all data intensive stuff lands
⋅⋅* data/datasets - various datasets we use for training our networks.

⋅⋅* data/givenModels - trained CNN models that are provided by outside sources.

⋅⋅* data/learnedModels - trained CNN models that we produce.

## Available Systems

<table>
  <tr>
    <td>Name</td>
    <td>IP address</td>
    <td>GPUs</td>
  </tr>
  <tr>
    <td>deepserver</td>
    <td>136.187.100.114</td>
    <td>1 GTX 960</td>
  </tr>
  <tr>
    <td>alienware</td>
    <td>N/A</td>
    <td>1 GTX 970M</td>
  </tr>
  <tr>
    <td>vm.psyche</td>
    <td>*ask around for details*</td>
    <td>2 GTX 980</td>
  </tr>
  
  <tr>
    <td>vm.selene</td>
    <td>*ask around for details*</td>
    <td>2 K40</td>
  </tr>
</table>

In _deepserver_, the `/home/shared` directory belongs to a special user account called shared, with an associated group to which all deep learning team members belong.

## Environment Setup on new systems / users

Follow these instructions to set up your development environment on a system.

1. Define your `.gitconfig` properly.

  Assuming you have installed `git` and have a [GitHub](github.com) account. Follow the guide [here](https://git-scm.com/book/en/v2/Getting-Started-First-Time-Git-Setup).

  If you are not familiar with Git(Hub) look for tutorials on Google, there is plenty of them, but you will learn by using it, just try not to do something that can delete information if you are not sure and ask around when in doubt.

  You most likely also want to add your `SSH` public key to GitHub to push and pull with no passwords though still secure, check [how](https://help.github.com/articles/adding-a-new-ssh-key-to-your-github-account/).

2. Clone repositories

  Clone this repo:

  ```bash
  git clone git@github.com:miquelmarti/DroneSegmentation.git
  ```

3. Add CUDA to your library path

  The machine learning framework Caffe is implemented using NVidia’s CUDA libraries, so we need to set up references to these libraries in your `.bashrc` file.  Run the following commands:

  ``` bash
  echo "export PATH=/usr/local/cuda/bin:$PATH" >> ~/.bashrc

  echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH" >> ~/.bashrc
  ```

4. Add locations to your PYTHONPATH

  In order for python scripts to make use of Caffe’s python libraries, the `PYTHONPATH` environment variable must be modified to include the path to the desired Caffe version’s python bindings. To set this variable to point to the default version of Caffe, run the following command:

  ``` bash
   export PYTHONPATH=/home/<username>/caffe/python:$PYTHONPATH
  ```

  In the case of _deepserver_, Caffe package is already available with the user `shared`.

  Now the python path will automatically include the Caffe python bindings when you login.  Again, you can change this line to point to a different Caffe version (such as that of SegNet) if you like.

  Furthermore, the `caffeUtils` python module which is used by much of our other code. For the other code to use this module, it must be in the `PYTHONPATH` variable as well.  So if you’ve cloned `DroneSegmentation` to your home directory (as will be described below), run the following command:

  ```bash
  export PYTHONPATH=/home/<username>/DroneSegmentation:$PYTHONPATH
  ```

  Finally, we must make sure that this `PYTHONPATH` variable will be preserved next time we log in.  Run the following command to add the proper export commands to your `.bashrc` file:

  ```bash
  echo "export PYTHONPATH=$PYTHONPATH" >> ~/.bashrc
  ```

  Remember that changes made in this file won’t take effect until you either log in again or source it:

  ```bash
  source ~/.bashrc
  ```

## Protocol Buffers

The Caffe framework makes extensive use of protocol buffers to define its configuration file formats, and our own code makes use of them as well.  You will probably find it worthwhile to read some of the documentation and study some of the examples, particularly as they relate to Python and to protobuf’s human-readable-text message format.

[https://developers.google.com/protocol-buffers/docs/overview](https://developers.google.com/protocol-buffers/docs/overview)

## **Tips & Tricks**

The topics in this section are optional but recommended, since they will probably make your life easier!

### Define Caffe alias

The different versions of the caffe tool used to train and test networks live in `/home/<username>/<caffe version>/build/tools`. So ordinarily, if you want to run the the default version of the tool, you’d have to execute a command like this:

```bash
 /home/shared/caffe/build/tools/caffe train <some arguments>
```
If you’re going to be frequently running the caffe tool to train and test networks, this will get annoying fast.  So to save typing, you can define an alias like this:

```bash
alias caffe="/home/shared/caffe/build/tools/caffe"
```
And then when you want to train a network you just have to type:

```bash
caffe train <some arguments>
```
Like the `PYTHONPATH` variable, this is something that will only last for the current shell session, so to make it permanent you can add it to your `.bashrc` file on its own line.  You can have separate aliases for different versions of Caffe if you like.

## Mount Network Drive

You can mount a directory on the server on your own computer, and edit the files with an editor running on your own machine, a much faster and more pleasant experience.

If your home machine is running Ubuntu or a similar linux distro, you can do this with the command:

```bash
gvfs-mount mount ssh://@136.187.100.73
<enter your username and password when prompted>
```

If you then look at your desktop, you’ll see a directory called `136.187.100.72`, which your computer’s GUI will treat like any other directory. Open it, navigate to the desired files, and edit them like you would any file on your home machine.

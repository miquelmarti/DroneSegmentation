# DroneSegmentation
Tools for training Caffe models for Semantic Segmentation of drone images

*outdated - file kept for future reference*

## Development System Architecture

Prendinger Lab has three Ubuntu servers that are used by the deep learning team for development and training of neural networks.  All three servers reside in the server room on the 15th floor.  se are:

<table>
  <tr>
    <td>Name</td>
    <td>IP address</td>
    <td>GPUs</td>
  </tr>
  <tr>
    <td>kai-i7</td>
    <td>136.187.100.72</td>
    <td>1 Titan X GPU</td>
  </tr>
  <tr>
    <td>kai-xeon</td>
    <td>136.187.100.73</td>
    <td>1 Titan X GPU</td>
  </tr>
  <tr>
    <td>kai-deeptitan</td>
    <td>136.187.100.75</td>
    <td>2 Titan X GPUs</td>
  </tr>
</table>

Various shared resources exist on each server under the directory /home/shared.  You can do an ls to see them all.  At the time of this writing, the most important shared directories are:

* caffe - the code and compiled binaries of the Caffe deep learning framework (usually kept at a more-or-less up-to-date version).

* caffeSegNet - the SegNet project, along with its special modified version of Caffe.

* hgRepos - various Mercurial repositories containing the code we’ve written.

* data/datasets - various datasets we use for training our networks.	

* data/givenModels - trained CNN models that are provided by outside sources.

* data/learnedModels - trained CNN models that we produce.

The /home/shared directory belongs to a special user account called shared, with an associated group to which all deep learning team members belong.  The password is not recorded in this document, but it is common knowledge among the team members, so just ask someone.

The /home/shared/data directory physically exists on kai-xeon, and is mounted as an ssh-filesystem to the same location on kai-i7 and kai-deeptitan.  Changes made under /home/shared/data on one server will immediately be reflected on the others.

# **Environment Setup**

Follow these instructions to set up your development environment on the server.

## 1. Add CUDA to your library path

The machine learning framework Caffe is implemented using NVidia’s CUDA libraries, so we need to set up references to these libraries in your .bashrc file.  Run the following commands:

#### $ echo "export PATH=/usr/local/cuda/bin:$PATH" >> ~/.bashrc

#### $ echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH" >> ~/.bashrc

## 2. Add locations to your PYTHONPATH

In order for python scripts to make use of Caffe’s python libraries, the PYTHONPATH environment variable must be modified to include the path to the desired Caffe version’s python bindings.  To set this variable to point to the default version of Caffe, run the following command:

#### $ export PYTHONPATH=/home/shared/caffe/python:$PYTHONPATH

Now the python path will automatically include the Caffe python bindings when you login.  Again, you can change this line to point to a different Caffe version (such as that of SegNet) if you like.  

Furthermore, the caffeUtils repo contains a python module which is used by much of our other code.  For the other code to use this module, it must be in the PYTHONPATH variable as well.  So if your username is andrew and you’ve cloned caffeUtils to your home directory (as will be described below), run the following command:

$ export PYTHONPATH=/home/andrew:$PYTHONPATH

Finally, we must make sure that this PYTHONPATH variable will be preserved next time we log in.  Run the following command to add the proper export commands to your .bashrc file:

$ echo "export PYTHONPATH=$PYTHONPATH" >> ~/.bashrc

Remember that changes made in this file won’t take effect until you either log in again or run this command:

#### $ source ~/.bashrc

## 3. Create your .hgrc file

This file defines some user preferences for our version control system, Mercurial.  If you don’t know anything about this file, there is a default one in /home/shared that you can use.  Copy it to your home directory:

#### $ cp /home/shared/.hgrc ~

Then edit your copy to include your real name and e-mail address:

#### $ gedit ~/.hgrc

#### # then set your name and e-mail appropriately

If you’re not familiar with Mercurial, it is highly recommended that you work through this short tutorial that explains how to use it:

[http://hgbook.red-bean.com/read/a-tour-of-mercurial-the-basics.html](http://hgbook.red-bean.com/read/a-tour-of-mercurial-the-basics.html) 

## 4. Clone repositories

The last thing to do is check out any code you want to work on.  As mentioned above, all of our code and network architecture files are stored in a series of Mercurial repositories in the directory /home/shared/hgRepos on kai-xeon.  You can ls this directory to see the current available repositories.  Feel free to clone any of these into your working directory.

# **Software Architecture and Repositories**

The work of the Deep Drone team primarily consists of experimenting with convolutional neural network architectures.  We obtain existing architectures from others and develop new ones ourselves.  We train networks with these architectures on a variety of datasets, and then measure the accuracy and time-performance of the networks.  Primarily the networks we work with perform semantic-segmentation - that is, they take an image as input, and output a meaningful label for each pixel of the image (for instance, whether that pixel is part of a cat, a dog, a tree, etc.)

For our work with convolutional neural nets (CNNs), we use the [Caffe](http://caffe.berkeleyvision.org/) framework.  Caffe is unique in that one can define and train a network without having to do any actual programming.  Caffe network architectures and training parameters are all defined in .prototxt configuration files which are fed to the Caffe binary, which trains a network and outputs the weights as a .caffemodel file.  If you’re not familiar with Caffe, it is highly recommended that you work through the below two tutorials to gain some familiarity with the system:

[http://caffe.berkeleyvision.org/gathered/examples/cifar10.html](http://caffe.berkeleyvision.org/gathered/examples/cifar10.html)

[http://caffe.berkeleyvision.org/gathered/examples/finetune_flickr_style.html](http://caffe.berkeleyvision.org/gathered/examples/finetune_flickr_style.html)

To expand on the basic features of Caffe, we have borrowed, altered, and written from-scratch a considerable amount of Python code that makes use of Caffe’s Python bindings.  This code is maintained in a number of Mercurial repositories, which are described below.

## caffeUtils

The caffeUtils repo is a python module that provide a variety utilities for deep learning with Caffe.  Most of these are based on code provided in [the FCN github project](https://github.com/shelhamer/fcn.berkeleyvision.org), but are heavily modified by us.  Make sure that you add the location of this repo to your PYTHONPATH environment variable, as described in the section "Environment Setup".  Details on the contained code can be found in the repo’s README.md file.

## caffeTools

The caffeTools repo is where we keep most of the scripts that we use to train, test, and evaluate Caffe networks, as well as to manipulate the datasets.  Most of the scripts here make use of the caffeUtils module.  Usage information for all of them can be found by running them with the -h option, and a more detailed description of their purpose and usage can be found in the repo’s README.md file.

## transferLearningFramework

This repo is where we contain our transfer-learning framework, a system for performing transfer-learning and multi-source training with Caffe.

* protobuf/transferLearning.proto: defines the config file format for describing a transfer-learning or multi-source learning job.

* scripts/transfer.py: the main script that takes a .prototxt config file (as defined in transferLearning.proto) as input, and carries it out.

* examples/: this directory contains various test cases and examples of configuration files for the transfer-learning framework.  Useful to study these to understand how to use this tool.

* configure: a simple bash script that auto-generates protocol-buffer python files from .proto message definition files.  When you first check out this repo, this script will need to be run before you can use some of the other scripts that rely on it.  You’ll also need to run it again if you make changes to any .proto files in this repo.

## models

This repo contains the various Caffe configuration files that define the network architectures and solving procedures that we use to train our models, as well as multi-source jobs we use (in the multiSource/ subdirectory).

## Protocol Buffers

The Caffe framework makes extensive use of protocol buffers to define its configuration file formats, and our own code makes use of them as well.  You will probably find it worthwhile to read some of the documentation and study some of the examples, particularly as they relate to Python and to protobuf’s human-readable-text message format.

[https://developers.google.com/protocol-buffers/docs/overview](https://developers.google.com/protocol-buffers/docs/overview)

# **Workflow**

To manage shared development on our code bases, we maintain a directory /home/shared/hgRepos, which contains a set of Mercurial repositories.  These repositories are meant to act as centralized hubs of development, which we regularly sync our changes with so that everyone can see them.  We centralize the repositories in this way to prevent the codebases of different people, or different subsets of people, from diverging over time.  These repositories are not meant to be modified directly, and so they are all "bare" - that is, they each contain only a Mercurial database, with no working copy of the files.  This part of the document describes the workflow we use to maintain synchronization.

## Working on an existing repository

Let's say you want to work on code in a repo (caffeTools for example).  First you need a local copy of caffeTools.  Check it out with:

#### $ hg clone ssh://136.187.100.73//home/shared/hgRepos/caffeTools


Or if you already have a local copy but haven't touched it in a while, cd into it and update it to be current with the central repository:

#### $ hg pull && hg update

 
You’ll then make some changes, and eventually you’ll decide you're happy with them and want to combine them with the central repo so everyone can see them.  After committing the changes in your local copy (with a helpful commit message), first synchronize your repo with the central one:

#### $ hg pull

Then, if necessary, merge your code with the results of the pull:

#### $ hg merge

If your changes conflict with some changes from the central repo, you may need to do some manual merging as well.  Once the merge is done, commit the results of the merge:

#### $ hg commit

Finally, once there are no conflicts, you can push to the central repo:

#### $ hg push


Next time someone does a pull from this repo, they’ll receive your changes.

## Creating a new repository

Sometimes you’ll start work on a project that doesn’t belong in any of the existing repositories.  Assuming you want to share this project with others, simply create a new Hg repository in the directory where you’re doing your work, and when it’s ready to share perform the following steps.

We’ll be making changes to the /home/shared directory on kai-xeon, so first switch to the shared user account (the password is the same as everyone’s) and navigate to the hgRepos dir:

#### $ ssh shared@136.187.100.73

#### $ cd hgRepos

Then make a clone of your own repository.  If your username is "andrew" and your project lives in a directory “myNewProject” inside your home dir, do the following:

#### $ hg clone /home/andrew/myNewProject

Since we keep all repositories in /home/shared/hgRepos as bare repos, let’s make this one bare too:

#### $ cd myNewProject && hg update null

# WARNING: Do NOT do a commit here!

Finally, we must modify the permissions of the repo so that everyone can push changes to it:

#### $ chmod g+s .hg .hg/store .hg/store/data

And that’s it!  Other people who want to work on your project should clone this repository to get started.  It should now be treated as the central, authoritative repo, and changes made to clones of it should be periodically synced back to it, as described in the previous section.

# **Tips & Tricks**

The topics in this section are optional but recommended, since they will probably make your life easier!

## Define Caffe alias

The different versions of the caffe tool used to train and test networks live in /home/shared/<caffe version>/build/tools.  So ordinarily, if you want to run the the default version of the tool, you’d have to execute a command like this:

#### $ /home/shared/caffe/build/tools/caffe train <some arguments>

If you’re going to be frequently running the caffe tool to train and test networks, this will get annoying fast.  So to save typing, you can define an alias like this:

#### $ alias caffe="/home/shared/caffe/build/tools/caffe”

And then when you want to train a network you just have to type:

#### $ caffe train <some arguments>

Like the PYTHONPATH variable, this is something that will only last for the current shell session, so to make it permanent you can add it to your .bashrc file on its own line.  You can have separate aliases for different versions of Caffe if you like.

## Mount Network Drive

If you try to edit files on the server with a proper GUI over SSH, you will likely find it quite slow, since all the GUI information is being passed over the SSH link.  To avoid this, you can mount a directory on the server on your own computer, and edit the files with an editor running on your own machine, a much faster and more pleasant experience.

If your home machine is running Ubuntu or a similar linux distro, you can do this with the command:

#### $ gvfs-mount mount ssh://@136.187.100.73

# enter your username and password when prompted

If you then look at your desktop, you’ll see a directory called "136.187.100.73", which your computer’s GUI will treat like any other directory.  Open it, navigate to the desired files, and edit them like you would any file on your home machine.

# **Notes**

## Remounting /home/shared/data after reboot

Currently, the /home/shared/data directory exists physically on kai-xeon, and is remotely mounted on kai-i7 and kai-deeptitan to the one on kai-xeon.  This is done using sshfs, and at the present time on kai-i7 and kai-deeptitan, it must be re-mounted manually whenever the machine reboots.  This must be done as the shared user like so:

$ su - shared

$ sshfs -o allow_other -o reconnect shared@136.187.100.73:data data

If need be, it can also be cleanly unmounted with:

$ fusermount -u /home/shared/data

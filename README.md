# Multi-Source Training

In neural networks, transfer learning refers to the practice of first training a net on one dataset, then taking some or all of the learned weights as initial weights for training on a second dataset, which may or may not have a similar distribution to the first.  Multi-source learning consists in performing transfer learning cyclically between two or more datasets for some number of iterations.  The below pseudocode defines it more precisely:

~~~python
# a neural-network training algorithm
# inputs: a dataset, and optionally a set of initial weights
# outputs: a set of trained weights
learn(D, W);
# num. of iterations to run multi-source training for
num_iterations;
# the initial weights for the learner, may be random
W_init;
# datasets to transfer between
datasets = [D_1, D_2, ..., D_n];

latest_weights = W_init
i = 0
while i < num_iterations:
    for D_i in datasets:
        latest_weights = learn(D_i, latest_weights)
    i += 1
return latest_weights
~~~

The output weights from each call to `learn(D, W)` are used as the input weights to the next call.  In theory, each time the dataset is changed, the solution space is perturbed, and a solution that has settled into a small, unstable local optimum on one dataset will find itself no longer in a local optimum for the next dataset.  This effectively perturbs the solution and helps it finally settle in a more globally-optimal solution.

This software package implements the multi-source training algorithm using the Caffe neural-network learning framework to carry out the learning.  Taking inspiration from Caffe, it allows the user to define a multi-source learning procedure simply by writing a config file and passing it to a generic interpreter, thus preventing the user from needing to write code.  Like Caffe, we use Google's protocol buffer system to define the configuration format.

# Usage

## Setup
run the configure executable (./configure). This will use the protocol buffer compiler to generate the file scripts/transferLearning_pb2.py, which is used by the framework's other python code.

## Defining a multi-source training job

At minimum, a multi-source training job consists of:
- a list of Caffe training jobs, called "stages", which will be carried out in the provided order.  Each stage description consists of:
  - the location of a Caffe solver.protoxt file that defines the Caffe job
  - optionally, a list of names of layers in the network to "freeze", setting the learning rates of those layers to 0.
  - optionally, a list of names of layers to "ignore", discarding the weights of those layers from the initial set of weights provided to this stage.
  
- the number of iterations, which indicates how many times the specified list of stages will be repeated.
- optionally, the input weights: the location of .caffemodel file containing a set of weights to use as the initial weights for the first stage in the first iteration of the multi-source job.

- optionally, a special *init stage*.  It may be that the format of the weights of the last stage differs from that of the input weights of the first stage at the first iteration.  In this case, we may wish for the first stage to freeze and/or ignore a different set of layers during the first iteration than those it does in subsequent iterations.  This can be acheived by specifying an "init stage".  This is a stage with the same format as the other stages, but during the first iteration of the multi-source job, Caffe will run this stage instead of the first stage in the stage list.  After the first iteration, it will be ignored, and the first stage in the provided list will be run at the start of each iteration.

To define a particular multi-source training job, we specify all of these parameters in a protocol-buffer text-format configuration file.  The format of these files is defined by the file protobuf/transferLearning.proto, and a variety of example configuration files can be found under examples/transfer_tests.  They all have the suffix ".prototxt".  New users should inspect both transferLearning.proto and the example configuration files to see how to define their own multi-source learning jobs.  An example is presented below.

    # A name for this multi-source learning job.
    name: "transfer_fcn"
    # The directory where the output weights will be saved.
    out_dir: "snapshots"
    # A set of weights to start from instead of initializing from nothing.
    init_weights: "/home/shared/data/givenModels/vgg-16/VGG_ILSVRC_16_layers.caffemodel"

    multi_source {
        # The number of times to iterate over the list of stages.
        iterations: 5

        # An optional init_stage is provided in this example.
        init_stage {
            name: "init"
            solver_filename: "../train_fcn/fcn32/solver.prototxt"
        
            ignore: "fc6"
            ignore: "fc7"
        
            fcn_surgery: true
            outLayer: "score"
        }

        # The list of stages to be carried out at each iteration.
        stage {
            name: "train_fcn-32"
            solver_filename: "../train_fcn/fcn32/solver.prototxt"
        
            fcn_surgery: true
            outLayer: "score"
        }
        stage {
            name: "train_fcn-16"
            solver_filename: "../train_fcn/fcn16/solver.prototxt"
        
            fcn_surgery: true
            outLayer: "score"
        }
        stage {
            name: "train_fcn-8"
            solver_filename: "../train_fcn/fcn8/solver.prototxt"
        
            fcn_surgery: true
            outLayer: "score"
        }
    }


## Running a multi-source training job

Once you've written a .prototxt file defining your multi-source training job, you'll want to execute the job.  This is done by passing the script as an argument to the script scripts/transfer.py:

`$ scripts/transfer.py ~/myMSJobs/transfer.prototxt`

Information on other optional arguments to this script can be found by running the script with the -h flag:

`$ scripts/transfer.py -h`

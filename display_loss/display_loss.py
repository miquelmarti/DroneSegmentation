# Takes, as inputs, the file with all the losses and the frequency of the display (display parameter in the solver.prototxt used for the training). Plot the loss then, with matplotlib.
# Use : python display_loss.py loss.txt 20


import argparse
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()

# Mandatory options
parser.add_argument('text_file_with_losses', type=str, help='Path to the file that lists all the loss')
parser.add_argument('display_frequency', type=int, help='Number between each displayed iteration')

args = parser.parse_args()


loss = [float(line.rstrip('\n')) for line in open(args.text_file_with_losses)]
x = np.arange(0, len(loss)*args.display_frequency, args.display_frequency)

plt.plot(x, loss)

plt.show()

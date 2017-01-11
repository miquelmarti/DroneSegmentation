# Dataset repository

Each dataset consists of different files and folders:

- version: name of the dataset version:
	  - split: name of the split (train, val, test)
        - images: folder containing the images
        - ground_truth: folder containing the ground truth label images
        - list.txt: file linking original images to ground truth label images
        - mean.txt: file with the per-channel mean of the dataset
- color_scheme: file assigning a color for each class for visualization of GT and results
- Class_list.txt: file assigning a name to each class number
- dataset explanation: TODO

List of datasets:

- Okutama
- Swiss

To get the datasets in the right format download the compressed files and  run the notebooks for each of them. If you want to add a new dataset, the easiest is to adapt the previous notebooks, which should not be much work.

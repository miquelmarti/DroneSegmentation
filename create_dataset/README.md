# Convert labels :
Used the colors to convert all the png of a given file (like train.txt) and save the nez png in the output folder.
Labels PNG input :  H x W x C (C = 3), corresponding to the color of the class
Labels PNG output : H x W x C (C = 3), corresponding to the number of the class
IMPORTANT : I finally realize that the labels of Pascal VOC 2012 have like two sides >>> When we open them with OpenCV, they display the label in RGB (with color of classes for each pixel) ; On the other hand, when we open them with Image (from PIL), they display the label in greyscale (with the number of the classes)>
So, no need to convert them, you just have to open the labels with Image (from PIL)


# Convert labels drone :
Used to convert all the png labels from Kai-Yan format (with the number of class in each pixel, but in the wrong format (RGB), like [0,4,2]) to the format used for LMDB (a simple number between 0 and nb_of_classes-1).
The last update was done in March the 30th, the number of classes and their number may have changed.


# Create dataset lmdb :
Create the dataset in LMDB format containing the images of the given text file (like train.txt). Labels flag is mandatory for creating lmdb groud truth datasets.


# Create drone colors :
An ad-hoc code generating the colors for the Drone dataset, all the colors are given arbitrarly in this code, all of them can be changed if needed.
Output : color.png, usable for visualization


# Find classes colors :
Used to generate the colors.png of a dataset.
color.png (1 x 256 x 3) :
[[  [0,0,0],
    [color of class 2],
    [color of class 3],
    ...,
    [color of class x],
    [0,0,0],
    ...
    [0,0,0] ]]




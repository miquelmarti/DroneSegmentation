Find classes colors :
Used to generate the colors.png of a dataset. The order of all the classes in the PNG is arbitrary, because there is usually no convention with this kind of dataset.
color.png (1 x 256 x 3) :
[[  [0,0,0],
    [color of class 2],
    [color of class 3],
    ...,
    [color of class x],
    [0,0,0],
    ...
    [0,0,0] ]]
IMPORTANT : Still in writing phase : The colors are not in the good order, I realize that it was a problem for the comparison with the ground truth (obviously). I have a code in /home/pierre/somewhere that correct the thing, but it's not perfect yet, and still have to tidy it


Convert labels :
Used the colors to convert all the png of a given file (like train.txt) and save the nez png in the output folder.
Labels PNG input :  H x W x C (C = 3), corresponding to the color of the class
Labels PNG output : H x W x C (C = 3), corresponding to the number of the class
IMPORTANT : I finally realize that the labels of Pascal VOC 2012 have like two sides >>> When we open them with OpenCV, they display the label in RGB (with color of classes for each pixel) ; On the other hand, when we open them with Image (from PIL), they display the label in greyscale (with the number of the classes)>
So, no need to convert them, you just have to open the labels with Image (from PIL)


Create dataset :
Create the dataset in LMDB format containing the images of the given text file (like train.txt). Labels flag is mandatory for creating lmdb groud truth datasets.

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


Convert labels :
Used the colors to convert all the png of a given file (like train.txt) and save the nez png in the output folder.
Labels PNG input :  H x W x C (C = 3), corresponding to the color of the class
Labels PNG output : H x W x C (C = 3), corresponding to the number of the class


Create dataset :
Create the dataset in LMDB format containing the images of the given text file (like train.txt). Labels flag is mandatory for creating lmdb groud truth datasets.

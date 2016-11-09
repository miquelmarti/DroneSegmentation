# Convert labels :
Used the colours or transitions to convert all the png of a folder and save the new pngs in the output folder.
Labels PNG input :  H x W x C (C = 3), corresponding to the color of the class OR to the number of the class (Kai format)
Labels PNG output : H x W x C (C = 3), corresponding to the number of the class

# Create dataset lmdb :
Create the dataset in LMDB format containing the images of the given text file (like train.txt). Labels flag is mandatory for creating lmdb groud truth datasets.

# Create transition :
Create the transition or colours from the text format.
output : xxx.png (1 x 256 x 3) :
[[  [0,0,0],
    [color of class 2],
    [color of class 3],
    ...,
    [color of class x],
    [0,0,0],
    ...
    [0,0,0] ]]

# Find classes colours :
Used to generate the colours.txt for pascal VOC. Only usable for pascal VOC for now, but would be available for each dataset that have the labels with the "double size" (it means ; cv2 for colors, PIL.Image for numbers)

# Get LMDB caracts :
Take a folder with the lmdbs as input and returns some basics caracteristics (shapes of images, format (RGB or Grey scale), number of images)

# Resize dataset :
Takes a folder with many images (labels or images) and resizes all of them in the good way.


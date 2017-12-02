Note public kaggle share copy last update 2017/12/02 

version 1: 
 - per class [A
 - conveted code to patch generators and improved the data agumentation with skew, rotate, noise 
 - failure points
  -- seems the trees class is what is diving the model to learn, maybe its the feature size and abundance that makes it wok

future
version2:
 - ROI detector, 
  -- lighter simple model that detects the presence of something.. then we apply the bigger unet model to mask it.. 
    this should reduuce over head as the ROI model will be much lighter, 
  -- also the amazon data set contains broad labeling.. 

future 2:
 - vgg net bottle neck roi model.. 

future 3:
 - progressive nets - train a lighter model the use its inputs enhance the inputs to the next version



undecided
 - data agumentation by cliping feature using its mask and transplanting it into another image, some logic is needed about what its layered in the images (ie cars above road or crops etc, structures in forest, crops, etc) 

 - ship detecting - using the san fan ship data from kaggle
 - better polygon normalization (this whole scale factor thing is just strange.. need to process the data so polygons are between [0,1] againest *image* not some magic numbers.. this will remove the silly parts of the data handling for polygons and standardize it for any new data added to the system
 - d3 roi marking.. sea of images with a rapid fire model auto selecting based on user action (adapt the existing one i have).. then map this classing back into a grid to auto formulate a polygon mask


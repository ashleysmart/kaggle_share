Note public kaggle share copy last update 2017/12/02 

###################### ACTION items ######################
- inspect patched training data.. confirm the reception size and training sanity
- roi_bottleneck model
- unet_bottleneck model
  -- kind of thinking i can do a unet model using the VGG/inception bottle necks taped at multiple stages

#################### VERSIONS / ROADMAP ####################

* version 1:
 - per class model ( ./per_class/ )
 - conveted code to patch generators and improved the data agumentation with skew, rotate, noise 
 - failure points
  -- seems the trees class is what is diving the model to learn, maybe its the feature size and abundance that makes it work

* version2:
 - ROI detector, ( ./roi_model/ )
  -- lighter simple model that detects the presence of something.. then we apply the bigger unet model to mask it.. 
    this should reduuce over head as the ROI model will be much lighter, 
  -- also the amazon data set contains broad labeling.. 
 - failure points
  -- models didnt train for the imbalanced classes...
   
version3:
 - ROI curves anaylsis / grid search ( ./roi_model/ )
 - implements a gird searcher/ learning curve system that can robustly trains and sample points in the model space   
 - failure points
   -- prep data size.. per class prep data means that i have 10x the input data... this is a total waste o disk space, and makes comparing across classes harder.. 
  -- temma has suggested to me(in a roundabout way: ie read chapter 7 of deeplearning book) that my test models 3 and greater are failing to train well due to poor regulization
  -- results are just awful (refer to ROIResults.ipynb )! i mark this up to:
   --- the large models where not training.. likely gradient/regulization issues?? 
   --- small models have receptive fields that are too small..
 - adding a regulized model didnt really improve the situation over all.. but im more intereasted in going to the bottlenecks models at the moment may come back here

version4:
 - vgg net bottle neck roi model.. 
  
#################### FUTURE STUFF ###########################

future:
 - progressive nets - train a lighter model, cross populate using its sibling models internal layer outputs(untrainable) as inputs to enhance models output

undecided
 - data agumentation by cliping feature using its mask and transplanting it into another image, some logic is needed about what its layered in the images (ie cars above road or crops etc, structures in forest, crops, etc) 

 - ship detecting - using the san fan ship data from kaggle
 - better polygon normalization (this whole scale factor thing is just strange.. need to process the data so polygons are between [0,1] againest *image* not some magic numbers.. this will remove the silly parts of the data handling for polygons and standardize it for any new data added to the system
 - d3 roi marking.. sea of images with a rapid fire model auto selecting based on user action (adapt the existing one i have).. then map this classing back into a grid to auto formulate a polygon mask

#################### LESSIONS LEARNED ###########################
* channels first order vs channel last order
 + GPU native order is channel first
 + advanced intel cpu instruction native order is channel first
 + channel first data loading is more efficenct.. ie loading multiple channels individually is in local and linear memory
 + https://github.com/fchollet/keras/issues/3149 - says channel first 20% faster
 + https://github.com/soumith/convnet-benchmarks/issues/66#issuecomment-155938969
- channel first images rendering required reordering for rgb renders...
 - tensorflow default is channels last...

 -->> RESULT
 - ChannelOrder.ipynb - results for channel first are piss poor.. channel last is 2-3x faster.. likely TF/keras overhead..

* data flow from disk
 - since this model is a patched based sampler i dont need to normalize the data if we flow it from disk.

* data formats
 - hdf5 data format... is basically the same size as npy ?? wtf why bother
 - npz using "save_compressed" has sizable reduction

* pre class models vs 1 multi class model
 + cross class population(multiple out classes) is critical for early layers developing features of a deep network 
 - the undersampled/difficult items are supported by the features created by the more simple items.. this gives them a leg up

* the importance of caching and process guarding executions
 - when producing learning curves its best to launch processes that create each long run cpu costly data point, this allows crashing and robustness 

* model design
 - when the roi model matches the leading shape from unet the unet model the results far exceed the ROI models
   -- this is due to the the gradient forwarding effects that created by the short cut links from the end/mid layers (its the same as resnets gradient bypass links)


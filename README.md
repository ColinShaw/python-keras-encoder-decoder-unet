# Vehicle Recognition

This is a collection of mechanisms for performing segmentation 
of image data with respect to cars.  The data source is Udacity's
[annotated driving dataset](https://github.com/udacity/self-driving-car/tree/master/annotations).
This data is used as a source of images and result masks
for training networks to directly map an input image to a 
car segmentation mask.  

The annotations from the datasets have some aspects that are
inconsistent and incorrectly labeled.  The two datasets have
different formats for the `.csv` files.  I added a header to 
the Autti dataset, replaced the space delimiters with commas
and removed the string delimiters from the label.  This dataset
also has sub-labels for traffic light color, so an extra header
column was added for that.  Both of the datasets had incorrect
identification of the bounds, with `xmax` and `ymin` swapped.
This is easy to identify by plotting bounding boxes on the 
original image.  If you intend to use this code with these 
datasets, you will need to udpate the `.csv` files for both
to meet these requirements. 

The approach I took to the masking was to generate the masks 
and resize the images in a completely separate step from training
the network.  The reason for this is I would prefer to simply have a 
collection of segmentation training data and not have to decode it 
each time it is run.  I also would like this training set for 
use in the future without having to remember that the bounding boxes
required correction.  For this problem, since there are 22065 total
images and I have a working available memory of 5.5GB on my 
GTX 1060, I decided to resize to 240x160.  This is a slightly different
aspect ration than the original image, but the actual scaled dimension
of 240x150 has problems with the max-pooling and upsampling in the model
due to divisibility.  For three-channel feature images and single-channel 
label segmentation masks, this is a total consumption of just under 3.4GB, 
leaving plenty of room for the model on the GPU.

For training, the resized images are loaded.  The feature images are 
left as RGB, whereas the label images are grayscale.  OpenCV loads our 
grayscale images as color, so we deliberately have to transform back
to grayscale.  The label is forced to an appropriate shape and normalized.
Note pre-initializing `numpy` arrays for loading, as this conserves memory
for large datasets.  

Further transformations are defined for the following operations, 
which are applied by the Keras generator:

 * Luminance to simulate different lighting conditions
 * Translation both horizontal and vertical to simulate different car positions
 * Expansion with unconstrained aspect ratio to simulate different car geometries

All of these are applied to the feature images, but only the geometric
transformations are applied to the label images.  The transformations
are applied in a Keras generator for augmentation.  The generator 
supports batches because batching allows us to train faster by not
making backpropagation steps for each feature/label pair.  One has to be
particularly careful with the dimension of tensors in the translation
and expansion.  The reason for this is that the masks are pre-normalized,
and when OpenCV performs operations on a single channel, the assumption is
made to not make the single-dimensional data abstract to multiple dimensions.
This is not how TensorFlow sees the world, so we have to reshape it.

The loss function used is the intersection over union measure.  Well, the 
negative of the intersection over union, as the function itself is a cost 
function. This simply measures the similarity of the two images by computing 
the relative overlap.  It is important to use the Keras backend methods for 
computing this quantity because the ultimate object of the predicted value is a 
TensorFlow Tensor object that does not admit various solutions (for example 
iteration).  The intersection over union figure is computed in the obvious 
way.

There are two candidate models, both encoder-decoders. The first model is simple 
and is able to be specified using a Keras `Sequential` object.  This type 
of encoder-decoder is fairly common.  The second model is what is called 
a `U-net` because in certain diagrams the model looks somewhat like a `U`.  It is
quite similar to the first model, but between similar convolutional layers 
there is a merge of the layers.  This has the effect of allowing deep convolution
layers to be merged with less deep convolution layers, and has been found to
help increase performance in segmentation problems ([U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)).  This model cannot be
implemented using a Keras `Sequential` object, because the merging is not 
sequential. 

You will note that the border mode is `same`.  Unlike a network where the top end
feeds fully connected layers, with both of these networks we are most interested in
preserving the layer size.  That is to say, the output prediction needs to be the 
same size as the input feature image in order for us to be able to compute our
intersection over union properly.  The simplest way to do this is to use `same` to
make the size under max-pooling and up-sampling invariant when using the same stride.
The reason I use `elu` is it is smoother than `relu`.  The reason this is important
to me is I want smooth transitions on the final segmentation.  While the activation 
function applies to the specific image elements, having an activation function
with smooth derivatives facilitates greater smoothness around the origin, which 
is what we are using to compute the intersection over union.  Consequently smoother
segmentation.

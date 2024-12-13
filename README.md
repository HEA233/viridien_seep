This project trains a U-Net model to detect “seep” regions in radar images. Each image is 256x256 pixels, and each pixel is labeled as either non-seep or one of several seep classes. U-Net is a common deep learning architecture for segmentation tasks—basically, it figures out which parts of the image belong to which category.

We use a training set of image/mask pairs and monitor progress with a loss function (lower is better) and the IoU metric (higher is better) to see how well predictions match actual seep areas.


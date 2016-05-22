# Eos
This project will allow raw image data to be processed into
tensorflow shards and then be used to train tensorflow models

The sample data used is the flower data from somewhere (source)

# To Do:
* Read in raw data
* Convert raw data to shards
* Abstract Base Class for model
* Driver to run everything
* Training module
* Image preprocessing
    * Live
    * Ahead of time (more drive space needed)
* Make a more complete readme
* Add in support for other image types (right now only jpeg)
# Deep learning based model for Cyro ET Sub-tomogram-Detection

High degree of structural complexity and practical imaging constraints make retrieval of
macromolecular structures from cryo-ET is very challenging. For image classification of
large-scale systematic macro-molecular structure from cryo-ET data.

For image classification of large-scale systematic macro-molecular structure from cryo-ET data, a
deep learning-based image classification approach has been employed to improve the
accuracy for a small range of SNR values where the present models have fallen short. 
Here, a novel SEC3 model for macro-molecule separation has been used. 

The model comprises 3D convolutional blocks and 3D squeeze and excitation blocks. The
model is trained on subtomogram datasets divided into 3 SNR values. Each SNR value
namely- 0.03, 0.05, and infinity is further divided into 10 different classes based on their
shape. The model understands by learning the valuable spatial information and
spectral information available in the macromolecular structure data set. Valuable
features detected by the convolution layers that are important for classification are
given importance, thereby improving the overall model accuracy.

Data  structure used for the project:

    new data
    |
    |--SNR003
    |    |
    |    |--1bxn
    |    |   |
    |    |   |--subtomogram_mrc 
    |    |   |--subtomogram_png
    |    |   |--densitymap_mrc
    |    |   |--densitymap_png
    |    |   |--json_label
    |    |   |--json_simulation
    |    |
    |    |
    |    |--1f1b, 1yg6, 2byu, 2h12, 2ldb, 3gl1, 3hhb, 4d4r, 6t3e
    |        |
    |        |...(the same as 1bxn)
    |    
    |
    |--SNR005
    |    |
    |    |...(the same as SNR003)
    |  
    |  
    |--SNR infinity
         |
         |...(the same as SNR003)

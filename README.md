# Image Classification Methods and Their Application to Own Handwritten Numbers

This project is **an overview of image classification methods**, performed at MNIST dataset. 
The purpose was to train and test different ML models at available digit images 
and then apply them to the **unseen data** -  larger numbers (unknown length) written by myself.

### Tested methods:
1) **Support Vector Machines** (SVM)
   - Linear kernel with parameters chosen via cross-validation 
achieved 0.921 accuracy on the test set,
   - After PCA (95% variance retained; 
   reducing the number of features allowed double 
   size of the training set), accuracy increased to 0.937.
2) **Multilayer Perceptrons** (MLP)
    for each model different sizes hidden layers was tested. 
    - the best performance for two-layers network was 0.976,
    - the best performance for three-layers network was the same.
   the simplest model with accuracy at least 0.97 was two-layers network with `hidden_size = 128`
3) **Convolutional Neural Networks** (CNN)
    - simply CNN was trained by 15 minutes and achieved accuracy 0.989 at test set,
    - training of the pretrained CNN (`MobileNet_v2`) was very computationally expensive:
   the first epoch took 1.5 hours and achieved an accuracy of 0.928.

### Application to new unseen data
1) Preprocessing of images
   - the images were collected from a tablet and manually cropped into rectangles and labelled. 
Next, the frequency of each digit (and the distribution of number lengths) were analyzed. (`data_labelling.ipynb` script)
   - first step of validation workflows was to prepare digit to the "mnist" format: 28x28 images with centered digits in grey scale:
     - split each number to digits (done correctly in 96.7% of cases),
     - lower the resolution,
     - adjust color scale,
     - blur, add the margins and resize result to required 28x28.
2) Selection of preprocessing parameters
    - using selected MLP model, preprocessing parameters was tuned 
   by grid search (`neural_networks_at_own_data.ipynb`),
    - performance of model at preprocessed data was evaluated: 
   best three-layers net achieved accuracy of 0.805 (for single digits) 
   and 0.512 (for whole numbers).
3) Test of simply CNN
   - simply CNN (`simply_cnn.ipynb`) was used to predict preprocessed numbers 
   and achieved accuracy of 0.927 for single digits and 0.783 for whole numbers. 

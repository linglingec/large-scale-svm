# Large Scale SVM

## Introduction
This readme file describes our directory for the Programming Assignment 2.

## Structure
The directory contains the following files:
-   `models.py`: contains our *LinearSVM* implementation
-   `optimizers.py`: contains our *MiniBatchSGD* and *Adagrad* implementation
-   `RFF.py`: contains our *RandomFourierFeatures* implementation
-   `utils.py`: contains our 'read data' function *get_data()*, *grid_search_svm()* and *cross_val_score()*
-   `test.ipynb`: is the final Jupyter Notebook where the assignment is done with the help of all latter implementations 
-   `PA2-Report.pdf`: is the corresponding report to this programming assignment

Further descriptions of classes and functions can be found in the respective files.
The directory includes also a data folder which contains the data (`imdb.npz`, `toydata_large.csv`, `toydata_tiny.csv`) needed to run the experiments.

## Important Note
Please note that especially the cells where the implementations are run on the `imdb.npz` are very time-consuming. We would not suggest running them on your own device
unless you have plenty of time.

## Any problems?
In case that you have problems with our directory or can't find something etc. feel free to reach out to one of us, we will be happy to help!
- Egor Fadeev: a12313685@unet.univie.ac.at
- Thomas Geier: a12026958@unet.univie.ac.at
# Zoidberg

Zoidberg2 is a project for classifying medical images of chest X-rays. The aim of this project is to develop a Deep Learning model capable of detecting chest abnormalities on medical images and classify them into two categories: "pneumonia" or "normal".

The project uses the Python programming language and libraries such as TensorFlow, Keras, Pandas, Scikit-learn, Seaborn, and Matplotlib for data processing, model construction and training, performance evaluation, and result visualization.

The Zoidberg2 project consists of several files, including a main file named zoidberg2.ipynb that contains the project's source code. This file is divided into different parts, each dedicated to a specific step in the model development process. The different parts of the zoidberg2.ipynb file are accompanied by comments and markdowns that explain their content and purpose.

### Clone the repository

```shell
cd some-project-folder
git clone git@github.com:EpitechMscProPromo2024/T-DEV-810-PAR_20.git
cd T-DEV-810-PAR_20
```

# Architecture

```
.
├── checkpoints                                  <- Checkpoints
├── saved_model                                  <- Trained models
│   └─── my_model
│        ├── variables
│        ├── fingerprint.pb
│        ├── saved_model.pb
│   └─── my_model1
│        ├── variables
│        ├── fingerprint.pb
│        ├── saved_model.pb
├── training_1                                   <- Training
│   ├── checkpoint
│   ├── cp-0001.ckpt.data-00000-of-00001
│   ├── cp-0001.ckpt.index
│   ├── cp-0002.ckpt.data-00000-of-00001
│   ├── cp-0002.ckpt.index
├── .gitignore
├── classify.py                                  <- file
├── README.md
├── zoidberg.ipynb                               <- Scripts to handle data (download, metrics,tensorflow)
├── zoidberg2.ipynb                              <- Scripts to handle data (download, metrics,tensorflow)
```

### Branch naming convention

- **feature**: zoidberg/feature/feature-name
- **fix**: zoidberg/fix/bug-name

## :books: Technologies

- [tensorflow](https://www.tensorflow.org/?hl=fr) (TensorFlow.js is an open-source library that allows developers to train, deploy, and run machine learning models in the browser using JavaScript. It is an extension of the TensorFlow library that was originally developed by Google.)
- [pandas](https://pandas.pydata.org/) ()
- [scikit-learn](https://scikit-learn.org/stable/) ()
- [pyyaml h5py](https://docs.h5py.org/en/stable/build.html) ()

# Zoidberg

Zoidberg is a project for classifying medical images of chest X-rays. The aim of this project is to develop a Deep Learning model capable of detecting chest abnormalities on medical images and classify them into two categories: "pneumonia" or "normal".

The project uses the Python programming language and libraries such as TensorFlow, Keras, Pandas, Scikit-learn, Seaborn, and Matplotlib for data processing, model construction and training, performance evaluation, and result visualization.

The Zoidberg project consists of several files, including a main file named zoidberg.ipynb that contains the project's source code. This file is divided into different parts, each dedicated to a specific step in the model development process. The different parts of the zoidberg.ipynb file are accompanied by comments and markdowns that explain their content and purpose.

### Clone the repository

```shell
cd some-project-folder
git clone git@github.com:EpitechMscProPromo2024/T-DEV-810-PAR_20.git
cd T-DEV-810-PAR_20
```

```
python -m venv myvenv

# windows
./myvenv/Scripts/activate
# ubuntu
source myvenv/bin/activate

pip install -r requirements.txt

flask --app web/main --debug run
```

# Architecture

```
.
├── checkpoints                                  <- Checkpoints
├── files_generated                              <- files generated
├── model_converted                              <- model converted
├── old_versions                                 <- old versions
├── saved_model                                  <- Trained models
│   └─── my_model
│        ├── best_model.h5
├── web
│   └─── static                                  <- HTML static for chest X-ray image classification web application.
│   └─── templates                               <- HTML template for chest X-ray image classification web application.
│   └─── main.py                                 <- Flask app for chest X-ray image classification using TensorFlow model.
├── .gitignore
├── README.md
├── requirement.txt
├── zoidberg.ipynb                               <- Scripts to handle data (download, metrics,tensorflow)
```

### Branch naming convention

- **feature**: feature/feature-name
- **fix**: fix/bug-name
- **doc**: doc/bug-name

### commit naming convention

- **feature**: feat: feature-name
- **fix**: fix: bug-name
- **doc**: doc: bug-name

## :books: Technologies

- [tensorflow](https://www.tensorflow.org/?hl=fr) (TensorFlow.js is an open-source library that allows developers to train, deploy, and run machine learning models in the browser using JavaScript. It is an extension of the TensorFlow library that was originally developed by Google.)
- [pandas](https://pandas.pydata.org/) (Pandas is a library written for the Python programming language allowing data manipulation and analysis)
- [scikit-learn](https://scikit-learn.org/stable/) (Scikit-learn is a free Python library for machine learning)
- [pyyaml h5py](https://docs.h5py.org/en/stable/build.html) (It is highly recommended that you use a pre-built version of h5py, either from a Python Distribution, an OS-specific package manager, or a pre-built wheel from PyPI.)

# SemEval Task 2 

## The Right Name Will Present Itself

### Venv Setup 

Project was created in python version `3.11.10` 

You can use `pyenv` to set an explicit version to use if you are on another version (will require downloading pyenv)
[https://github.com/pyenv/pyenv]{PyEnv Github}

> Somethings may be fine if on another python version I just have no clue honetly 

Create a virtual environment with `python -m venv abc` (this creates a venv in the current directory and names the venv folder `abc`)

Activate the venv by typing `source ./abc/bin/activate` on mac (honestly not sure how it works on windows)

Install required packages via `pip install -r requirements.txt`

### Acknowledgements

The evaluation script for this project was created using the template provided by Apple for the SemEval Task 2. The template can be found [here](https://github.com/apple/ml-xc-translate/tree/main/evaluation).


## Usage

In order to get started with this repository ensure you are on python version `3.11` and create a virtual environment. Install dependencies from our `requirements.txt` file.

### Code base
A majority of our codebase can be found in the `src` folder where we implemented a lot of our models and datasets. However some of the code for implementing the pipelines can be found in the `notebooks` folder.


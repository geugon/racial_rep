# racial_rep
Congressional racial representation based on district demographics - a simple Keras model with Streamlit


## Overview
This is a quick tool to estimate the probability of a congress person's ethnicity based soley on the ethnic distribution of the district they represent.  The aim of this model is provide information to better understand when over/under representation occurs and the effect of redistricting upon racial represetation.


## Setup and Run
```bash
conda create -n py38 python=3.8 keras ipykernel pandas
conda activate py38  # may be "source activate" if using newer conda version
pip install streamlit
streamlit run display.py 
```


## Implementation
A secondary motivation underlying this work is demonstration of a small end-to-end project for educational purposes.  As such, additional description of the methods are provided.

#### Data Preparation
Data was obtained from https://docs.google.com/spreadsheets/d/1GNRnHy677PD8Je6Rp3evaJ5ty5MbCIDIAkSJVJ3ax2c
Equivalently well-organizd input was not found for previous years, so the overall results could possibly be improved by additional background work in finding data sources.

Racial breakdown of each district divided into 6 catagories: White, Black, Hispanic, Asian, Native, and Other in the original data.  Data is normalized on a 0 to 1 scale matching the population fraction for each catagory and the "Other" data is dropped, as it is a simple linear combination of the remaining data.  All other data, including the party afflications, previous election results, etc. is ignored.  See Methodology section for why.

The ethnicity of each congress member is mapped to those same catagories as best as possible, except that no "Other" congressional representative exists.  This is the target data and is converted to a one-hot representation.

#### Model Creation and Training
As interpretability is a major consideration of this work, a single-layer neural network classification is performed.  This is equivalent to a logistic regression with multiple outputs.  While other packages are likely capable of implementing this model, Keras is used.  This serves as a minimal example of how to use Keras.

No validation or testing sets were created and used.  This is normally a very poor choice.  See Methodology section for further discussion.  Otherwise, training is a straight-forward process of using the fit method of a Keras model.

#### Deployment
Streamlit provides the ability to rapidly create a minimal interface.  This deployment option as the lowest learning curve to creating a clear representation of model ouptus.  Additionally, relevent graphs are easily embedded.

Behind the interface itself, a previously trained version of the model is loaded.  All of this happens locally, but could be adapted to run on a hosted webpage with some additional effort.


## Methodology and Impact
pending completion of code


## Results and Discussion
pending completion of code

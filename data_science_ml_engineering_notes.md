###  Welcome to the Zero to Mastery Data Science and Machine Learning Bootcamp!

* ml-playground.com/#

## Create a framework

1. Problem definition - What problem are we trying to solve
  a. Supervised Learning - I know my inputs and outputs
    1. Does a patient have heart disease Yes or No? Classification
    2. What is the avg value of home? Regression
  b. Unsupervised Learning - I'm not sure of the outputs but I have inputs
    1. Which customers are similar based on their purchase history - Clustering
  c. Transfer Learning - I think my problem may be similar to something else
    1. Can I use an existing machine learning model has learned to train my own?
  d. Reinforcement Learning - Playing Chess

2. Data (More Data the Better!) - What data do we have?
  Static or Streaming Data
  a. Structured Data
   1. Excel/CSV spreadsheets, Rows by Columns
  b. Unstructured Data
   1. Images, audio files

3. Evaluation - What defines success?
 a. Classification
  1. Accuracy - >95%
  2. Precision
  3. Recall
 b. Regression
  1. MAE
  2. MSE
  3. RMSE
 c. Recommendation
  1. Precision at K
 
4. Features - What do we already know about the data/What features should we model?
 Want > 10% coverage
 a. variables - Weight,Sex,Heart Rate, Chest pain
  1. Numerical features
  2. Categorical features
  3. Derived feature - Feature engineering
 
5. Modelling - What kind of model should we use?
 3 sets - Training(70%-80%), Validation(10%-15%), Testing (10%-15%)
 Generalization
 a. Choosing a model
  1. Structured Data - CatBoost, XGBoost, Random Forest
  2. Unstructured Data - Deep Learning, Transfer Learning
 b. Train a model
  1. Use features X(data) to find y(labels)
  2. Goal is to minimize time between experiments
 c. Tuning the Model - Validation Data
  1. hyperameters - use to tune
   a. Random Forest - adjust no of trees
   b. Neural Networks - adjust no of layers
 d. Model comparison - Test Data
  1. Underfitting - Test Data is under Training Data
   a. Data mismatch - Different features in Test Data and Training Data
   b. Fixes - try more advanced model
  2. Overfitting - Test Data is over Training Data
   a. Data leakage - Like cheating on final exam
   b. Fixes - try less complicated model
  3. Balanced - (Goldilocks zone)
  4. Things to remember
   a. Avoid overfitting and underfitting (head towards generality)
   b. Keep test set separate all costs

**** All experiments should be conducted on different portions of your data.

Training data set — Use this set for model training, 70–80% of your data is the standard.
Validation/development data set — Use this set for model hyperparameter tuning and experimentation evaluation, 10–15% of your data is the standard.
Test data set — Use this set for model testing and comparison, 10–15% of your data is the standard.
These amounts can fluctuate slightly, depending on your problem and the data you have.

Poor performance on training data means the model hasn’t learned properly and is underfitting. Try a different model, improve the existing one through hyperparameter or collect more data.

Great performance on the training data but poor performance on test data means your model doesn’t generalize well. Your model may be overfitting the training data. Try using a simpler model or making sure your the test data is of the same style your model is training on.

Another form of overfitting can come in the form of better performance on test data than training data. This may mean your testing data is leaking into your training data (incorrect data splits) or you've spent too much time optimizing your model for the test set data. Ensure your training and test datasets are kept separate at all times and avoid optimizing a models performance on the test set (use the training and validation sets for model improvement).

Poor performance once deployed (in the real world) means there’s a difference in what you trained and tested your model on and what is actually happening. Ensure the data you're using during experimentation matches up with the data you're using in production.****
 
6. Experimentation - How could we improve/what can we try next?
 a. Adjust inputs
 b. Adjust model
 c. Update Outputs.


## Match to data science and machine learning tools.

1. Anaconda & Jupyter Notebooks - Install on PC
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
./Miniconda3-latest-Windows-x86_64.exe

conda --version

conda create --prefix ./env pandas numpy matplotlib scikit-learn

ml-course/sample-project/env
ml-course/sample-project/env
miniconda3

conda /env

cd "/c/ZeroToMastery/AI_ML_Engineer/ml-course/sample_project"
cd "/c/ZeroToMastery/AI_ML_Engineer/ml-course/"

echo "# zerotomastery-ml-course" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/johnkdevops/zerotomastery-ml-course.git
git push -u origin main

ln -s "/c/ZeroToMastery/AI_ML_Engineer/DataScienceAndMLBootcamp/" ml-course


conda activate env/ && jupyter notebook

jupyter notebook

alias activate_sample='conda activate "/c/KodeKloud_Pro/Google Cloud/Google Cloud Platform Engineer/zerotomastery/AI ML Engineer/DataScienceAndMLBootcamp/sample_project/env"'

Git Private Repo: /c/ZeroToMastery/AI_ML_Engineer/ml-course (main)

conda env list

conda install jupyter

# Start up Jupyter notebook
jupyter notebook

conda deactivate

#Share your Conda Environment
conda env export > environment.yml
conda env create --prefix ./env -f ../sample-project/environment.yml
conda env create --prefix ./env pandas numpy matplotlib jupyter scikit-learn

Dog Breed Project Data (Deep Learning Project)

!wget https://www.dropbox.com/s/9kjr0ui9qbodfao/dog-breed-identification.zip # download files from Dropbox as zip

import os
import zipfile

local_zip = 'dog-breed-identification.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')

zip_ref.extractall('Dog Vision') # unzip the files into a file called "Dog Vision"
zip_ref.close()

Bulldozers Project Data (Milestone Project 2)

!wget https://github.com/mrdbourke/zero-to-mastery-ml/raw/master/data/bluebook-for-bulldozers.zip # download files from GitHub as zip

import os
import zipfile

local_zip = 'bluebook-for-bulldozers.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')

zip_ref.extractall('.') # extract all data into current working directory
zip_ref.close()


```

Sure, I will be your buddy.  I am currently taking Data Science/ML Mastery Course.

2. Data Analysis tools
 a. pandas
  1. Simple to use
  2. Integrated with many other data science & ML Python tools
  3. Helps you get data ready for ML
 b. matplotlib
 c. NumPy
3. Machine Learning tools
 a. TensorFlow
 b. PyTorch
 c. sckitlearn
 d. CatBoost
 e. XGBoost

```bash
alias ztm='cd '/c/KodeKloud_Pro/Google\ Cloud/Google\ Cloud\ Platform\ Engineer/zerotomastery'

cd "/c/KodeKloud_Pro/Google Cloud/Google Cloud Platform Engineer/zerotomastery"



```
## Learn By Doing



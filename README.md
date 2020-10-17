# Adthena_assessment

## Installation

1. Install Python 3.6.5

https://docs.anaconda.com/anaconda/install/

2. Clone this repository

3. Run the command below in the terminal:

**pip install -r requirements.txt**

## Training and testing the model

To train the model using a new dataset run the command below in the terminal. 

**python train.py your_dataset.csv**

Accuracy and loss metrics are shown in the command line as the model is being trained. Model pre-trained on trainSet.csv can be found in the link below:

https://drive.google.com/drive/folders/1MJx0QjNRbzzFjuyR3PnlT6EIL_CwMmZx?usp=sharing

## Prediction

Once you have a pre-trained model called **model.h5** ou can run the command below to make predictions:

**python predict.py you_prediction_file.txt**

The predictions will be saved as a dataframe called testSet.csv

import pandas as pd

"""
This program is used to separate the data between training data and evaluating data.
- training set : to train the model. 80%
- validation set : is used during training to test the model. 20%

Data rows are made of an id number, the diagnosis and the measurements.
The id is not useful for the agent to learn and could be detrimental if it learns based on it instead of the measurements data.

212 M
357 B
Total = 569

30 data variable

cmd : python3 data_divider.py
"""

df = pd.read_csv('data.csv', header=None)
print(df.shape)
df = df.drop([0], axis=1)
limiter = round(df.shape[0] * 0.8)
training_set_df = df.iloc[:limiter,:]
validation_set_df = df.iloc[limiter:,:]
print(training_set_df.shape)
print(validation_set_df.shape)
training_set_df.to_csv('training_data.csv', index=False, header=None)
validation_set_df.to_csv('validation_data.csv', index=False, header=None)

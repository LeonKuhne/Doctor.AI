import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# read in data
medical_data = pd.read_sas("data/RXQ_RX_I.xpt")
food_data = pd.read_sas("data/DR1IFF_I.xpt")

# combine into one table
data = pd.merge(medical_data, food_data, left_on="SEQN", right_on="SEQN")
#data.sort_values('RXDDRUG')

# extract columns
# DR1CCMTX - combination food type
# RXDDRUG - perscription medications
#food_data = data['DR1CCMTX']
#drug_data = data['RXDDRUG']

# convert categories to id's
food_ids = data.groupby('DR1CCMTX').ngroup()
drug_ids = data.groupby('RXDDRUG').ngroup()

# convert to numpy arrays
x = food_ids.values
y = drug_ids.values

print(x)
print(y)

# format y
y_matrix = to_categorical(y)

# create test data
x_train, x_test, y_train, y_test = train_test_split(x, y_matrix, random_state=7, test_size=0.2)

# determine num inputs and outputs
num_foods = len(np.unique(x))
num_drugs = len(np.unique(y))

# create the model
model = Sequential()
model.add(Dense(32, input_shape=(num_foods,), activation='relu'))
model.add(Dropout(0.5)) # prevent overfitting
model.add(Dense(num_drugs, activation='softmax'))

# compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 



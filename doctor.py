import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
from tensorflow.keras import metrics

# config
INIT_DENSITY = 1000
FOOD_FILE = "DR1IFF_I"
drug_question = 'RXDDRUG'
food_questions = ["DR1IGRMS", "DR1IKCAL", "DR1IPROT", "DR1ICARB", "DR1ISUGR", "DR1IFIBE", "DR1ITFAT", "DR1ISFAT", "DR1IMFAT", "DR1IPFAT", "DR1ICHOL", "DR1IATOC", "DR1IATOA", "DR1IRET", "DR1IVARA", "DR1IACAR", "DR1IBCAR", "DR1ICRYP", "DR1ILYCO", "DR1ILZ", "DR1IVB1", "DR1IVB2", "DR1INIAC", "DR1IVB6", "DR1IFOLA", "DR1IFA", "DR1IFF", "DR1IFDFE", "DR1ICHL", "DR1IVB12", "DR1IB12A", "DR1IVC", "DR1IVD", "DR1IVK", "DR1ICALC", "DR1IPHOS", "DR1IMAGN", "DR1IIRON", "DR1IZINC", "DR1ICOPP", "DR1ISODI", "DR1IPOTA", "DR1ISELE", "DR1ICAFF", "DR1ITHEO", "DR1IALCO", "DR1IMOIS", "DR1IS040", "DR1IS060", "DR1IS080", "DR1IS100", "DR1IS120", "DR1IS140", "DR1IS160", "DR1IS180", "DR1IM161", "DR1IM181", "DR1IM201", "DR1IM221", "DR1IP182", "DR1IP183", "DR1IP184", "DR1IP204", "DR1IP205", "DR1IP225", "DR1IP226"]
MIN_THRESHOLD = 0.00001

# setup tensorboard
tensorboard = TensorBoard(log_dir=f"logs/doctor-{int(time.time())}")

# read in data
medical_data = pd.read_sas("data/RXQ_RX_I.xpt")
food_data = pd.read_sas(f"data/{FOOD_FILE}.xpt")

# combine into one table
data = pd.merge(medical_data, food_data, left_on="SEQN", right_on="SEQN")
select_columns = np.concatenate(([drug_question], food_questions), axis=0)
data = data[select_columns]
data = data.dropna() # remove nan values
#data.sort_values('RXDDRUG') # sort data

print(data)

# extract columns
# RXDDRUG - perscription medications
#food_data = data[FOOD_QUESTION]
drug_data = data[drug_question]
#drug_ids = data.groupby('RXDDRUG').ngroup()
drug_categories = drug_data.values

def get_food_question_arr():
    maxs = []
    x = data[food_questions].values

    for col_id in range(len(food_questions[0])):
        col_data = x[col_id]

        # replace weird low value with 0
        col_data[col_data < MIN_THRESHOLD] = 0
 
        # make between 0-1 with max
        question_max = np.max(col_data)
        maxs.append(question_max)
        x[col_id] = col_data/question_max

    print(f"question maxs: {maxs}")
    return x

def get_ant_acids_bool_arr():
    # y categories to antacid true/false
    y_arr = []
    antacids = ['omeprazzole', 'pantoprazole', 'esomeprazole', 'lansoprazole', 'ranitidine', 'cimetidine', 'famotidine']
    for drugs_encoded in drug_categories:
        drugs = drugs_encoded.decode("utf-8")
        contains = False
        for drug in drugs.split("; "):
            if(drug.lower() in antacids):
                contains = True
        y_arr.append(contains)

    y = np.asarray(y_arr)
    print(f"percent taking drug: {np.sum(y)/len(y)}")
    return y

# get vals
x = get_food_question_arr()
y = get_ant_acids_bool_arr()

print("shapes")
print(x.shape)
print(y.shape)

# format drugs to categories
y_matrix = to_categorical(y)

# create test data
x_train, x_test, y_train, y_test = train_test_split(x, y_matrix, random_state=7, test_size=0.2)

# determine num inputs and outputs
#num_drugs = len(np.unique(y))

# create the model
model = Sequential()
model.add(Dense(INIT_DENSITY, input_shape=(len(food_questions), ), activation='relu'))
for i in range(20):
    model.add(Dropout(0.05)) # prevent overfitting
    model.add(Dense(15, activation='relu'))

model.add(Dense(2, activation='softmax'))

# custom metrics

# compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'mse'])

# train
model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test), callbacks=[tensorboard])

# save
model.save("doctor.model")

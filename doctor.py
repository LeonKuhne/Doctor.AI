import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.callbacks import TensorBoard

# config
FOOD_FILE = "DR1IFF_I"
food_questions = ["DR1IGRMS", "DR1IKCAL", "DR1IPROT", "DR1ICARB", "DR1ISUGR", "DR1IFIBE", "DR1ITFAT", "DR1ISFAT", "DR1IMFAT", "DR1IPFAT", "DR1ICHOL", "DR1IATOC", "DR1IATOA", "DR1IRET", "DR1IVARA", "DR1IACAR", "DR1IBCAR", "DR1ICRYP", "DR1ILYCO", "DR1ILZ", "DR1IVB1", "DR1IVB2", "DR1INIAC", "DR1IVB6", "DR1IFOLA", "DR1IFA", "DR1IFF", "DR1IFDFE", "DR1ICHL", "DR1IVB12", "DR1IB12A", "DR1IVC", "DR1IVD", "DR1IVK", "DR1ICALC", "DR1IPHOS", "DR1IMAGN", "DR1IIRON", "DR1IZINC", "DR1ICOPP", "DR1ISODI", "DR1IPOTA", "DR1ISELE", "DR1ICAFF", "DR1ITHEO", "DR1IALCO", "DR1IMOIS", "DR1IS040", "DR1IS060", "DR1IS080", "DR1IS100", "DR1IS120", "DR1IS140", "DR1IS160", "DR1IS180", "DR1IM161", "DR1IM181", "DR1IM201", "DR1IM221", "DR1IP182", "DR1IP183", "DR1IP184", "DR1IP204", "DR1IP205", "DR1IP225", "DR1IP226"]

# setup tensorboard
tensorboard = TensorBoard(log_dir=f"logs/doctor-{int(time.time())}")

# read in data
medical_data = pd.read_sas("data/RXQ_RX_I.xpt")
food_data = pd.read_sas(f"data/{FOOD_FILE}.xpt")

# combine into one table
data = pd.merge(medical_data, food_data, left_on="SEQN", right_on="SEQN")
#data.sort_values('RXDDRUG')

# extract columns
# RXDDRUG - perscription medications
#food_data = data[FOOD_QUESTION]
drug_data = data['RXDDRUG']
#drug_ids = data.groupby('RXDDRUG').ngroup()
drug_categories = drug_data.values

def get_food_question_arr():
    # add all questions together
    food_question_maxs = []
    x = data[food_questions].values

    # normalize between 0 and 1
    for col_id in range(len(food_questions[0])):
        question_max = np.max(x[col_id]) 
        food_question_maxs.append(question_max)
        x[col_id] /= question_max
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
    return y


# get vals
x = get_food_question_arr()
y = get_ant_acids_bool_arr()

print("shapes")
print(x.shape)
print(y.shape)

# format x - make between 0 and 1


# format drugs to categories
y_matrix = to_categorical(y)

# create test data
x_train, x_test, y_train, y_test = train_test_split(x, y_matrix, random_state=7, test_size=0.2)

# determine num inputs and outputs
#num_drugs = len(np.unique(y))

# create the model
model = Sequential()
print(len(food_questions))
model.add(Dense(32, input_shape=(len(food_questions), ), activation='relu'))
for i in range(5):
    model.add(Dropout(0.5)) # prevent overfitting
    model.add(Dense(20, activation='relu'))

model.add(Dense(2, activation='softmax'))

# compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# train
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), callbacks=[tensorboard])

# save
model.save("doctor.model")

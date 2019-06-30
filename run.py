import argparse
import numpy as np
from keras.models import load_model
from keras.utils import to_categorical

# parse arguments
parser = argparse.ArgumentParser(description='"i am god" - carl')
parser.add_argument("-f", "--food", help="int representing food type")
args = parser.parse_args()

# select a combination food type (DR1CCMTX)
x = 0
if args.food:
    x = int(args.food)
else:
    print("must use -f [food]")
    exit(0)

MODEL_NAME = "doctor.model"
model = load_model(MODEL_NAME)


for x in range(99):
    prediction = model.predict(np.asarray([x]))
    print(f"{x}: {prediction}")


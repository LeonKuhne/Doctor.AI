import argparse
import numpy as np
from keras.models import load_model
from keras.utils import to_categorical

# parse arguments
parser = argparse.ArgumentParser(description='doctor says hi')
parser.add_argument("-c", "--count", help="total number of questions")
args = parser.parse_args()

# select a combination food type (DR1CCMTX)
count = 0
if args.count:
    count = int(args.count)
else:
    print("must use -c [int]")
    exit(0)

MODEL_NAME = "doctor.model"
model = load_model(MODEL_NAME)

profile = []
prediction = model.predict(np.asarray(profile))
print(f"{x}: {prediction}")


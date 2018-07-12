import pandas as pd
import numpy as np
import sys

if len(sys.argv) != 3:
    print("Argument Error!")
    exit(1)
    
print("PREDICTION_FILE=" + sys.argv[1])
print("TEST_FILE=" + sys.argv[2])
prediction = pd.read_excel(sys.argv[1])
truth = pd.read_excel(sys.argv[2])

df = pd.merge(prediction, truth, left_on='hour', right_on='hour', how='outer', sort='hour')
df.columns = ['hour', 'pred', 'truth']

score = 0
for idx, row in enumerate(df.values):
    h, p, t = row
    if np.isnan(p):
        continue
    if p == 1 and t == 1:
        score += 2
    elif p == 0 and t == 0:
        score += 1
    elif p == 1 and t == 0:
        if 1 in df.loc[idx-2:idx+2].truth.values:
            score += 0
        else:
            score -= 1
    elif p == 0 and t == 1:
        if 1 in df.loc[idx-2:idx+2].pred.values:
            score += 0
        else:
            score -= 1

print("Score = {}".format(score))
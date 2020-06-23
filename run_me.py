import numpy as np 
import numpy.fft as fft
import matplotlib.pyplot as plt 
import os
import sklearn
import librosa
from scipy import stats
from scipy import signal
from scipy.signal import find_peaks
from label import Label
from featureExtract import featureExtract
from sklearn import neighbors
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
import time


############################
############################
def convertLabel(k):
    if k == 0:
        lab='Stationary'
    elif k == 1:
        lab='Walking-flat-surface'
    elif k == 2:
        lab='Walking-up-stairs'
    elif k == 3:
        lab='Walking-down-stairs'
    elif k == 4:
        lab='Elevator-up'
    elif k == 5:
        lab='Elevator-down'
    elif k == 6:
        lab='Running'
    else:
        k = -1
    return lab

def showTable(header, body):
    fig = go.Figure(data=[go.Table(header=dict(values=header),
                 cells=dict(values=np.array(body).T))
                     ])
    fig.show()

############################
##************************##
############################


me = 'aaleksanyan' # When we're in the folder that matches 'me', set the student parameter in the loop to zero

subjectDirs = os.listdir('allData') ## Contain all directory names of the subjects
#subjectDirs = ['jbosworth', 'aaleksanyan'] #-- uncomment for smaller subdir subset to test with

windowSize = 100
windowShift = 16
start = time.time()
featureMatrix = featureExtract(subjectDirs, windowSize, windowShift, me)
featureMatrix = np.array(featureMatrix)

print("featureMatrix: ", featureMatrix.shape)
print("---- took %s seconds to assemble ----" %(time.time() - start))

## Okay, yay! Our feature matrix is populated and ready to party.
## Now we need to first split the feature matrix as such:
## We need the rows that are me (0) and the rows that are not me (!0)

me = []
notMe = []

for row in featureMatrix:
    if row[-1] == 0:
        me.append(row[:-1]) # Take out subjectID since we no longer need it
    else:
        notMe.append(row[:-1])

me = np.array(me, dtype=np.float64)
notMe = np.array(notMe, dtype=np.float64)        
print("Me v notMe shapes:", me.shape, notMe.shape)

# Train KNN Classifier
k = 5
knn = neighbors.KNeighborsClassifier(k)

# Train random trees
rfc = RandomForestClassifier()

for model in [(knn,'knn'), (rfc,'rfc')]:
    print("Creating classifier: ", model[1])
    clf, name = model
    clf.fit(notMe[:, :-1], notMe[:, -1:].ravel())

    # Precit outputs with classifier, save actual outputs
    ypredict = clf.predict(me[:, :-1]).flatten()
    yactual = me[:, -1:].flatten()

    # For each activity, calculate precision, recall and f-score. Create a table.
    header = ["Activity - " + name, "Precision", "Recall","F-Score"] 
    body = []

    report = classification_report(yactual, ypredict, output_dict=True)

    for act in range(7):
        asStr = str(float(act))
        p = report[asStr]['precision']
        r = report[asStr]['recall']
        f = report[asStr]['f1-score']
        body.append([convertLabel(act), p, r, f])

    showTable(header, body)

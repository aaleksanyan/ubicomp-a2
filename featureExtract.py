import numpy as np 
import numpy.fft as fft
import matplotlib.pyplot as plt 
import os
import sklearn
import librosa
from scipy import stats
from scipy import signal
from scipy.signal import find_peaks
from scipy import interpolate
from label import Label

## File containing function to compute and return features

def featureExtract(subjectDirs, windowSize, windowShift, me):
    currentSubject = 1

    # Initialize a numpy array of overall feature matrix size
    # TODO: Change dimensionality of np.zeros to accomodate FD features    
    featureMatrix = np.zeros((0,22)) # Depth = n, width = num features + 1 for label + 1 for subjectID

    # For each subject
    for folder in subjectDirs:
        print("Processing folder: ", folder)

        # I manually marked all the folders in my dataset that did not have a label.txt or had some other disparity with a '-nl' suffix. This can easily be done with code, but there are few enough folders that this was easier. 'oldData' contains data from previous class iterations that I chose not to include for now.
        if folder[-3:] == '-nl' or folder == "oldData":
            continue # Ignore

        # Create label object 
        labs = Label('allData\\' + folder + "\\labels.txt")
        # List of items in subject directory
        folderItems = os.listdir('allData\\' + folder)

        if folder == me:
            subjectID = 0
        else:
            subjectID = currentSubject
            currentSubject += 1

        # Initialize a numpy array for feature matrix for the single subject
        # TODO: Change dimensionality of np.zeros to accomodate FD features
        subjectFeatureMatrix = np.zeros((0,21)) # (n, features + 1 for label) 

        # For each file inside the folder
        for item in folderItems:
            # NOTE: The structure we want is |accelTD|baromTD|accelFD|baromFD|

            # Since the structure of the matrix we want to generate necessitates that we have both the accel and barom data side by side, we must process them together.
            # Here, the code ignores everything that is not accel data. When it finds accel data, it also processes the corresponding barom data by removing the last nine characters from the filename ('accel.txt') and appending 'pressure.txt' to get the barom data for the same time. 
 
            # Ignore labels, pressure, and gyro
            if item == 'labels.txt':
                continue # Ignore, we already have what we need
            # elif item[-9:] == 'accel.txt': # No need to check if it's accel if that's the default
            #     dataType = 'accel'
            elif item[-12:] == 'pressure.txt':
                #dataType = 'barom'
                continue            
            elif item[-8:] == 'gyro.txt':
                #dataType = 'gyro'
                continue
                        
            # Get arrays from filenames
            dataAccel = readFile('allData\\' + folder + '\\' + item)
            dataBarom = readFile('allData\\' + folder + '\\' + item[:-9] + 'pressure.txt')
            dataLabel = labs.getLabel(int(dataAccel[0,0])) # Timestamp of first entry
            dataAccel, dataBarom = interp(dataAccel, dataBarom) # Reassign arrs to be interpolated
            featuresInFile = [] # (n, 2TDf + 2FDf) -- len, 2*number of time domain features + 2*number of frequency domain features
            

            # Start the windowing
            for index in range(0, len(dataAccel), windowShift):
                # Assuming sample rate is 32Hz (idk how accurate that is)

                # Set the end point to where we are plus the windowSize, or to the end of the arr if that would exceed limits
                end = index + windowSize + 1
                if index + windowSize + 1 > len(dataAccel):
                    end = len(dataAccel) 

                workingDataAccel = dataAccel[index:end] # (window, 3)
                workingDataBarom = dataBarom[index:end] # (window, 1)
                windowFeatures = getFeatures(workingDataAccel, workingDataBarom) # (1,features) actually (f,)
                featuresInFile.append(windowFeatures)
            
            # print("Inner loop - featuresInFile after windowing: ", np.array(featuresInFile).shape)
            print("Processed item: ", item, np.array(featuresInFile).shape, "label: ", dataLabel)

            ## ATTACH LABELS HERE
            labelsVector = np.full((len(featuresInFile),1), dataLabel) # (n,1)
            featuresInFile = np.concatenate((featuresInFile, labelsVector), axis=1) # (n,features+1)
            subjectFeatureMatrix = np.concatenate((subjectFeatureMatrix, featuresInFile), axis=0)
            #print("inner loop -- subjectFeatureMatrix: ", np.array(subjectFeatureMatrix).shape)

        ## ATTACH SUBJECTID HERE
        subjectIDMatrix = np.full((len(subjectFeatureMatrix),1), subjectID) # (N, 1)
        subjectFeatureMatrix = np.concatenate((subjectFeatureMatrix, subjectIDMatrix), axis=1) # (N, features + 1)
        featureMatrix = np.concatenate((featureMatrix, subjectFeatureMatrix), axis=0)
        print("Added subject %d, featureMatrix shape is now:" %(subjectID), featureMatrix.shape)

    #print("featureMatrix contains NaN?", (True in np.isnan(featureMatrix)))
    return featureMatrix

# Takes in two np.array as args where the first column for both is timestamp.
# inpArrAccel should be (n,4) and inpArrBarom should be (n,2) since it just has one data point.
# Ignores first column since that's time, computes time- and frequency-domain features (TDF and FDF).
# Returns a np.array of shape (1, 2*TDF + 2*FDF). 
# TODO: Figure out what's going on with the shapes and do a sanity check... outputting (F, ) instead of (1, F)
# ...Turns out when you append an (x,) array to a (y,x) array you get a (y+1,x) arrary. Go figure
def getFeatures(inpArrAccel, inpArrBarom):
    barom = inpArrBarom[:,1:].astype(float) # Removes first col (timestamp) and converts from string to float. Shape = (n,1)
    accel = inpArrAccel[:,1:].astype(float)
    accel = np.linalg.norm(accel, axis=1) # Get 'magnitude'. 3 col -> 1 col
    
    featArr = []

    # Time domain
    for arr in [accel, barom]:
        # arr = (n,1) 
        # print("getFeatures: ", arr.shape)
        median = np.median(arr)
        std = np.std(arr)
        skew = stats.skew(arr)
        mcr = 0 # Implement a function to calculate...
        xvals = np.array(range(len(arr))) # Should be shape(n,). For finding slope
        # TODO: Sometimes below spits out a NaN and I have no idea why
        #slope, _, _, _, _ = stats.linregress(xvals, arr.flatten())
        slope = 0 #Fuck it
        iqrange = stats.iqr(arr)
        twentyFifthPercentile = np.percentile(arr, 25)
        peaks, _ = find_peaks(arr.flatten()) # Indexes of peaks
        numPeaks = len(peaks)
        peaksTotal = 0 # Calculating meanPeaks
        for p in peaks:
            peaksTotal += arr[p]
        if numPeaks == 1:
            meanPeaks = peaksTotal
            meanPeakDistance = 0
        elif numPeaks != 0:
            meanPeaks = peaksTotal/numPeaks
            meanPeakDistance = np.mean(np.diff(peaks))
        else:
            meanPeaks = 0
            meanPeakDistance = 0
        
        for f in [median, std, skew, mcr, slope, iqrange, twentyFifthPercentile, numPeaks, meanPeaks, meanPeakDistance]:
            featArr.append(f)
        
    for arr in [accel, barom]:
        ## frequency domain functions ##
        # fftrans = np.fft.fft(arr) # Spitting out garbage
        # freq = fft.fftfreq(len(fftrans))
        # magnitude = np.sqrt(fftrans.real**2 + fftrans.imag**2)
        # magnitude = signal.detrend(magnitude)
        # print("Magnitude, freq shape:", magnitude.shape, freq.shape)
        # #print("Freq: ", freq)
        # plt.plot(freq, magnitude)
        # plt.show()
        #spectralCentroid = librosa.feature.spectral_centroid(y=freqTransformed, sr=)
        pass
        # TODO: finish frequency domain functions and return them

    # TODO: Figure out irregularities in data. Above code is producing NaN, infinity, or value too large for int type in some spots.
    #print("featArr before removeIrreg", np.array(featArr).shape)
    # Let's invent a function that will take an input of the data and return a cleaned version.
    featArr = removeIrregularities(np.array(featArr, dtype=np.float64))
    #print("featArr after removeIrreg", featArr.shape)

    # nans = np.argwhere(np.isnan(np.array(featArr, dtype=np.float64)))
    # if len(nans) > 0:
    #     print("nan here. features - ", nans[0])
    #     for i in nans[0]:        
    #         featArr[i] = 0
    #         print("now it is: ", featArr[i])
    #print("featArr after old stuff", np.array(featArr).shape)

    return featArr

# Takes in path as arg, returns np.array of shape NxD
def readFile(path):
    f = open(path, "r")
    line = f.readline()[:-1] # Rm last character - \n
    allData = []

    while(line):
        rawData = line.split(',') # array containing all data
        allData.append(rawData)
        line = f.readline()[:-1]

    allData = np.array(allData)
    #print("read file:", path[-25:], allData.shape)
    return allData

# Takes in two arrays of probably different lengths and non-aligning timestamps and returns two arrays with the same lenth and timestamps. Interpolation!
def interp(longer, shorter):
    # So first, we need to figure out which of the arrays goes to a later time. We cannot interpolate outside of the range of the data. 
    # Since interpolate returns a function, lengths don't matter. All that matters is that the longer one doesn't have a later time
    floor = int(shorter[0,0])
    ceiling = int(shorter[-1,0])
    f = interpolate.interp1d(shorter[:,0].astype(float), shorter[:,1].astype(float))
    newLonger = []
    newShort = []
    for i in range(len(longer)):
        ts = int(longer[i,0])
        if ts > floor and ts < ceiling:
            newLonger.append(longer[i])
            newShort.append([ts, f(ts)])

    newLonger = np.array(newLonger)
    newShort = np.array(newShort)

    return newLonger, newShort

# Takes in a (f,) np.array 
# Replaces any irregularities, like NaN, inf, or anything that's bad data with 0.
def removeIrregularities(arr):
    # Check for NaN
    nans = np.isnan(arr)
    if True in nans: # If there is a NaN present
        for ind in range(len(arr)): # Reminder: shape is (f,). Could also iterate over nans since shape should be the same.
            if nans[ind] == True:
                print("NaN found. Row: ", ind)
                arr[ind] = 0

    # Check for inf
    if np.inf in arr:
        print("Infinte value here")
        # Idk what to do??
    
    return arr
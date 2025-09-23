
import matplotlib.pyplot as plt 
import mne 
from pyedflib import highlevel
import pyedflib as plib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.decomposition import PCA
import numpy as np
from pywt import wavedec
from sklearn.model_selection import train_test_split
import math
from os import listdir
from sklearn.model_selection import GridSearchCV 
from os.path import isfile, join
import glob
import ntpath
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


def preprocessSignals(raw, notchFreq, bandLowFreq, bandHighFreq):
    raw.notch_filter(freqs=notchFreq)
    raw.filter(l_freq = bandLowFreq, h_freq = bandHighFreq)

    signals = raw.get_data()

    return signals


#suskirstau po 120 langu skaiciu ir po 23 kanalus

def splitToWindows(signals, channelNumber, windowNumber, oneWindowSignalsNumber):

    #define  towdemensional array
    window = [[None for _ in range(channelNumber)] for _ in range(windowNumber)]

    for i in range(windowNumber):
        start = i * oneWindowSignalsNumber
        end = (i + 1) * oneWindowSignalsNumber
        for j in range(channelNumber):
            window[i][j] = signals[j, start:end]

    return window

#funckijos =======================================================================================================================================

def dwt_coeffs(signalVector):
    return wavedec(signalVector, 'db4', level=5)


# Compute absolute energy of coefficients
def absoluteEnergy(dwtCoeffs, numberOfDWTCoeffs, frequency):


    samplingInterval = 1 / frequency
    rate = (samplingInterval / numberOfDWTCoeffs)
    result = np.sum((dwtCoeffs*dwtCoeffs) * rate )

    return result

def relativeEnergy(abEnergyCoef, sumOfAbEnergy):
    return abEnergyCoef/sumOfAbEnergy

#https://stackoverflow.com/questions/3843017/efficiently-detect-sign-changes-in-python
#gaunami indeksai, kur kerta
def findZeroCrossings(coefs):
    zero_crossings = np.where(np.diff(np.sign(coefs)))[0]
    return zero_crossings

#gaunami segmentai
def findHalfWaveSEgments(a,zero):
    ApSeg = []
    start = 0
    end = zero[0] + 2
    for i in range(len(zero)-1):
        ApSeg.append(a[start:end])
        start = zero[i]+1
        end = zero[i+1]+2

    ApSeg.append(a[start:])
    return ApSeg

#gaunmos amplitudes
def findHalfWAveAmplitudes(ApSeg):
    halfWaveAmplitudes = []
    for ar in ApSeg:
        max = np.max(ar)
        min = np.min(ar)
        halfWaveAmplitudes.append(max - min)
    return halfWaveAmplitudes

def AEpoch(N,halfWaveAmplitudes):
    N= len(halfWaveAmplitudes)
    totalAEpoch = np.sum(halfWaveAmplitudes)
    Aepoch = totalAEpoch/N
    return Aepoch

def calculateAmplitude(Coeff):
    findZeroCrossingsCoeff= findZeroCrossings(Coeff)

    CoeffHalfWaveSegment = findHalfWaveSEgments(Coeff,findZeroCrossingsCoeff)

    findHalfWAveAmplitudeCoeff = findHalfWAveAmplitudes(CoeffHalfWaveSegment)

    AEpochBackground = AEpoch(len(findHalfWAveAmplitudeCoeff),findHalfWAveAmplitudeCoeff)

    return AEpochBackground

# kai backgroup signalas yra 120 sec o pertrauk 60 sec, tai tada bus 20 langu kanale

def splitToWindowsChosenSizeForAmplitudeBackgroundCalculating(channelNumber, frequency, signals, duration, gap):
    start = 0
    end = int(duration * frequency)

    # how many windows fit in the signal
    windowNr = int(3600 / (duration+gap))

    windows = [[None for _ in range(channelNumber)] for _ in range(windowNr)]
    indexes = []

    for w in range(windowNr):
        for ch in range(channelNumber):
            windows[w][ch] = signals[ch, start:end]

        # save (start, end) indices in samples
        indexes.append((start, end))

        # move to next window: add duration + gap
        backgroundDuration = int((duration + gap) * frequency)
        start = start + backgroundDuration
        end = start + int(duration * frequency)

    return windows, indexes, windowNr


def ABackgroundInDurationSignals(channelNumber, windows):
        listWindow=[]
        backgroundAmplitudes=[]
        background=[]
        for w in range(len(windows)):
            for ch in range(channelNumber):
                    cA5, cD5, cD4, cD3, cD2, cD1 = dwt_coeffs(windows[w][ch])
            
                    listWindow.append((cD3, cD4, cD5))
                    D3 = listWindow[ch][0]
                    D4 = listWindow[ch][1]
                    D5 = listWindow[ch][2]

                    AEpochD3Background = calculateAmplitude(D3)
                    AEpochD4Background = calculateAmplitude(D4)
                    AEpochD5Background = calculateAmplitude(D5)
                    

                    backgroundAmplitudes=[AEpochD3Background,AEpochD4Background,AEpochD5Background]
                    background.append(backgroundAmplitudes)

        return background

    
#viso signalo pilno amplitudziu apdorojimas, straipsnyje paimta kas 120 sec background, o mes meginsime bendrini paimti, nes kartais gali tu atitraukimu neuztekti hjeigu signalas trumpesnis
# todel eisime universalesniu budu
def ABackgroundInAllSignals(channelNumber, signals):
        listWindow=[]
        backgroundAmplitudes=[]
        background=[]
        for ch in range(channelNumber):
              cA5, cD5, cD4, cD3, cD2, cD1 = dwt_coeffs(signals[ch])
      
              listWindow.append((cD3, cD4, cD5))
              D3 = listWindow[ch][0]
              D4 = listWindow[ch][1]
              D5 = listWindow[ch][2]

              AEpochD3Background = calculateAmplitude(D3)
              AEpochD4Background = calculateAmplitude(D4)
              AEpochD5Background = calculateAmplitude(D5)
              

              backgroundAmplitudes=[AEpochD3Background,AEpochD4Background,AEpochD5Background]
              background.append(backgroundAmplitudes)

        return background

def coefficientVariation(coeffs, N):
    meanValue = (1/N) * np.sum(coeffs)
    standartDaviation = math.sqrt((1/N)*np.sum((coeffs-meanValue) * (coeffs-meanValue)))
    coefVariation = math.pow(standartDaviation,2)/ math.pow(meanValue,2)
    return coefVariation


#Kadangi nevienodi koeficientu dydziai reikia suvienodinti, todel nukerpam iki reikiamo dydzio, prepend tam, kad nenukirptu per daug 
def fluctuationIndex(coeff, N):


    fluctuationInd = (1/N) * np.sum(abs(np.diff(coeff,prepend=0)))
    return fluctuationInd



#FEATURE CALCULATION
#gaunu suma 1698 koeficientu (D3,D4,D5), KUR D3 966, D4 486 d5 246  viename lange viename kanale, turiu 23 kanalus, o langu gaunasi 120 
def featureCalculation(signals, windows, frequency, channelNumber,seconds):
    list = []
    features = []

    backgroundWindows, indexes, windowNr = splitToWindowsChosenSizeForAmplitudeBackgroundCalculating(channelNumber, frequency,signals, duration=120, gap=60)

    #backgroundWindows = np.array(backgroundWindows)   
    #print("Background windows shape :", backgroundWindows.shape) #(20, 23, 30720) # 20 langu, 23 kanalai ir 30720 verciu per kanala ir per lanag
    back = ABackgroundInDurationSignals(channelNumber, backgroundWindows)
    backValues = np.array(back)
    print("background amplitude values shape: ", backValues.shape) #(460,3) 20 langu 23 kanalai ir per viena kanala ir langa 3 koeficienta
    backValues = backValues.reshape(windowNr, channelNumber, 3)
    index = 0
   
    # I BUDAS IMTI BACKGROUND PER VISA 3600 SEC SIGNALA
    #background amplitude through all windows
    #background = ABackgroundInAllSignals(channelNumber,signals)
    for w in range(len(windows)):
        time = w*seconds
        listWindow = []
        listWindow2=[]
        listWindowAll=[]
        for ch in range(channelNumber):
            # Wavelet decomposition
            cA5, cD5, cD4, cD3, cD2, cD1 = dwt_coeffs(windows[w][ch])
            listWindow.append((cD3, cD4, cD5))
            listWindowAll.append((cD1,cD2, cD3, cD4, cD5))

            # RANDAME VISU KOEFICIENTU SKAICIU
            suma = len(cD3) + len(cD4) + len(cD5)
            sizeD3 = len(cD3)
            sizeD4 = len(cD4)
            sizeD5 = len(cD5)

            D3 = listWindow[ch][0]
            D4 = listWindow[ch][1]
            D5 = listWindow[ch][2]
            
            
            absoluteED3 = absoluteEnergy(D3, sizeD3, frequency)
            absoluteED4 = absoluteEnergy(D4, sizeD4, frequency)
            absoluteED5 = absoluteEnergy(D5, sizeD5, frequency)

            sumofAbsoluteEnergies = absoluteED3 + absoluteED4 + absoluteED5
            relativeED3 = relativeEnergy(absoluteED3, sumofAbsoluteEnergies)
            relativeED4 = relativeEnergy(absoluteED4, sumofAbsoluteEnergies)
            relativeED5 = relativeEnergy(absoluteED5, sumofAbsoluteEnergies)


            AEpochD3 = calculateAmplitude(D3)
            AEpochD4 = calculateAmplitude(D4)
            AEpochD5 = calculateAmplitude(D5)

            #background amplitude througha all windows and signals        
            #ABackgroundD3 = background[ch][0]
            #ABackgroundD4 = background[ch][1]
            #ABackgroundD5 = background[ch][2]

            #II BUDAS kadanghgi vienas langas apima 4 sec tai [0,120 sec] backgroyund verte galios iki pat 180 sec

            
            if time > 0 and time % 180 == 0:
              index = index + 1
                 #koreguoti
              index = min(index, windowNr - 1) 

            # III BUDAS nuo o iki 120 epochai galios [0,120], o nuo 120 epochos iki 300 galios [180, 300], nuo 300 iki 460 galios [360, 480]

            #if time > 120:
             #    index = int((time - 120) / 180) + 1
              #   index = min(index, windowNr - 1) 

          
            #if time < 120:
            #    index = 0
            #else:
            #    index = int((time - 120) / 180) + 1
            #    index = min(index, windowNr - 1)


          
            ABackgroundD3 = backValues[index][ch][0]
            ABackgroundD4 = backValues[index][ch][1]
            ABackgroundD5 = backValues[index][ch][2]
           
            ARelativeD3 = AEpochD3/ABackgroundD3
            ARelativeD4 = AEpochD4/ABackgroundD4
            ARelativeD5 = AEpochD5/ABackgroundD5

            coefficientVariationD3 = coefficientVariation(D3, sizeD3)
            coefficientVariationD4 = coefficientVariation(D4, sizeD4)
            coefficientVariationD5 = coefficientVariation(D5, sizeD5)

        
            FI3 = fluctuationIndex(D3, sizeD3)
            FI4 = fluctuationIndex(D4, sizeD4)
            FI5 = fluctuationIndex(D5, sizeD5)



            feature = [relativeED3, relativeED4, relativeED5,  ARelativeD3, ARelativeD4, ARelativeD5, coefficientVariationD3, coefficientVariationD4, coefficientVariationD5,FI3,FI4,FI5]
            features.append(feature)

        list.append(listWindow)
    

    features = np.array(features)
    print("Features shape:", features.shape)  # now (2760, 12)
    return features


#patobulinimas
# PIRMO priepolio pradzios indeksas 16, pabaiga 16.8. Antro priepolio pradzia 81.7, pabaiga 82.5. Pasiaiskinti kaip geriau daryti nes dabar vienas priepolis tarsi tampa kaip du priepoliai
def fillYWithSeizures( windowNumber, seizure_intervals, seconds):

    y=[]
    y = np.zeros(windowNumber)

    for startSeizureTime, endSeizureTime in seizure_intervals:

     
        start = int(math.floor(startSeizureTime / seconds))
        end = int(math.ceil(endSeizureTime / seconds))
        #end +1 tam, kad paimtu paskutini elementa
        y[start:(end+1)] = 1


    return y

def getXAndYAtSpecialIdexesWhenNONSeiziure(windowNumber, seizure_intervals, seconds,signals, channelNumber,oneWindowSignalsNumber, seizureWindowNUmberInFile):
     
     y = fillYWithSeizures(windowNumber,seizure_intervals,seconds)
     X = splitToWindows(signals, channelNumber, windowNumber, oneWindowSignalsNumber)

     
     seizureIndexes = []
     for index in range(len(y)):
       if(y[index] == 0):
         seizureIndexes.append(index)
  

     randomSeizuresIndexes = np.random.choice(seizureIndexes, size=seizureWindowNUmberInFile, replace=False)

     XRandomNonseizure = []
     for i in randomSeizuresIndexes:
            XRandomNonseizure.append(X[i])

     yRandomNonSeizure = y[randomSeizuresIndexes]
     return XRandomNonseizure, yRandomNonSeizure


def getXAndYAtSpecialIdexesWhenSeiziure(windowNumber, seizure_intervals, seconds,signals, channelNumber,oneWindowSignalsNumber):
     
     y = fillYWithSeizures(windowNumber,seizure_intervals,seconds)
     X = splitToWindows(signals, channelNumber, windowNumber, oneWindowSignalsNumber)

     seizureIndexes = []
     for index in range(len(y)):
       if(y[index] == 1):
         seizureIndexes.append(index)
    

     XSeizures = []
     for i in seizureIndexes:
        XSeizures.append(X[i])

     ySeizures = y[seizureIndexes]

     print("seizure index: ", seizureIndexes)
     return XSeizures, ySeizures

# kiek ivyko priepoliu kiek is 1 buvo nuejimu i 0
def CalculateNumberOfSeizures(y):

    y = np.array(y).flatten()   

    difList = np.diff(y)

    count=0
    for diff in difList:

        if diff==-1:
            count=count+1
    return count

#def prepareForSVMTraining(train_nr,seizures,channelNumber,X,y):
#     
#    XTrain = []
#    YTrain = []


    #training function
#    for i in range(train_nr):
#            
#        if seizures[i] > 0:
#            print("index:", i)
#            print("y simple:", y[i].shape) #(120)
#            print("X simple:", X[i].shape) #(2760,12 vectors)
#
#            y_train = np.repeat(y[i], channelNumber) #(2760)
#
#            XTrain.append(X[i])   # shape: (windows*channels, features)
#            YTrain.append(y_train)
#
#    XTrain = np.concatenate(XTrain)        
#    YTrain = np.concatenate(YTrain)  
#
#    return XTrain, YTrain 

       
#iprastai priepolis laikomas kai buna bent viename kanale 1 gera vieta postprocessing
def predictedYPostprocessing(Ymatrix):

    yPredictedWindow = []
    for i in range (len(Ymatrix)):
            #jei bent 2 kanalu toje eiluteje pasikartojo priepolis tada vadinasi tikrai buvo priepolis
            if Ymatrix[i].sum() >= 2:
                yPredictedWindow.append(1)
            else:
                yPredictedWindow.append(0)

    yPredictedWindow = np.array(yPredictedWindow)


#paskui jeigu kanale vienas priepolis tikriname kas buvo gretimuose segmententuose, jeigu nors vienas gretimas tada irgi uzskaitom kaip priepoli
# eisim iki priespaskutinio, nes tam, kad neislipti is ribu, nes ziuresim jo gretima kaimyna
    lengthTillSecondToLastElement = len(Ymatrix)-1
    for z in range (lengthTillSecondToLastElement):
            if Ymatrix[z].sum() >= 1 and ((z > 0 and yPredictedWindow[z-1] == 1) or yPredictedWindow[z+1] == 1):
                yPredictedWindow[z] = 1

    return yPredictedWindow


def collarTechnique(yPredictedWindow, collar):
    for i in range(len(yPredictedWindow) - 1):
        if yPredictedWindow[i] == 1 and  (i > 0 and yPredictedWindow[i-1] == 0):
            end = i
            # what if index is negative
            if i - collar > 0: 
                start = i - collar
            else:
                start = 0

            yPredictedWindow[start:end] = 1

        if yPredictedWindow[i] == 1 and yPredictedWindow[i+1] == 0:
            start = i
            end = (i + collar) + 1
            yPredictedWindow[start:end] = 1
    return yPredictedWindow
    

#SVM and accurancy and so on
#https://stackoverflow.com/questions/52217151/how-can-we-give-explicit-test-data-and-train-data-to-svm-instead-of-using-train
#https://www.kaggle.com/code/quangtranvo/lab-5-machine-learning-on-eeg-signals


def calculateMetrics(TP, FP, FN, TN):
    if (TP + FN) > 0:

        sensitivity = TP / (TP + FN)  
        
    else: 
         sensitivity = np.nan
    if (TN + FP) > 0:

        specificity = TN / (TN + FP)

    else: 
         specificity = 0.0
    if (TP + TN + FP + FN) > 0:
        accurancy = (TP + TN) / (TP + TN + FP + FN)

    else: 
         accurancy = 0.0
    if TP > 0:
    
        precision = TP / (TP + FP) 
    else: 
         precision = np.nan
    if (FP + TN) > 0: 
         
        fpr = FP / (FP + TN)  
    else: 
         fpr= np.nan
    return sensitivity, specificity, accurancy, precision, fpr




def MAFCalculation(signal, totalWindowNumber, channelNumber, windowSize):

    windowsRecreatedSize = totalWindowNumber - windowSize + 1
    columns = channelNumber
  
    result = [[None for _ in range(channelNumber)] for _ in range(windowsRecreatedSize)]

    for w in range(windowsRecreatedSize):     
        for ch in range(channelNumber):      
            start = w
            end = w + windowSize
            window = signal[start:end, ch]
            result[w][ch] = np.mean(window)

    return result



def calculateSVM(XTest, yTest, XTrain, yTrain, channelNumber, seizures, patientsNumber, windowNumber, train_nr):

    scaler = StandardScaler()

    #svc_clf = SVC(C=10, gamma='scale', kernel='rbf', class_weight="balanced")

    np.set_printoptions(threshold=np.inf)

    # scale train features
    # https://www.geeksforgeeks.org/machine-learning/svm-hyperparameter-tuning-using-gridsearchcv-ml/
    XTrainScaled = scaler.fit_transform(XTrain)
   

    param_grid = {'C': [0.1, 1, 10, 100, 1000], 
			'gamma': ['scale', 'auto', 1, 0.1, 0.01, 0.001, 0.0001], 
			'kernel': ['rbf','linear']} 

    grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3, n_jobs=-1, scoring='f1') 
 
    grid.fit(XTrainScaled, yTrain)
    #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

    print("Best parameters:", grid.best_params_)
    print("Best precision score:", grid.best_score_)
    svc_clf = grid.best_estimator_

    # Train SVM
    #svc_clf.fit(XTrainScaled, yTrain)

    trainAccurancy = svc_clf.score(XTrainScaled, yTrain)
    print("SVM TRAINING ACCURANCY: ", trainAccurancy)

    # Testing loop through all same patients 
    sens = []
    spec = []
    acc = [] 
    prec = [] 
    fpriii=[]

    sumTN = 0
    sumFN = 0
    sumFP = 0
    sumTP = 0


    for i in range(patientsNumber):
        
        XTestPerPatient = XTest[i]          # shape: (120*23, 12)
        YTestPerPatient = yTest[i]          # shape: (120,)


        # Apply scaling for X test
        XTestScaled = scaler.transform(XTestPerPatient)
        #X_test_i_pca = pca.transform(X_test_i_scaled)

        rawSignal = svc_clf.decision_function(XTestScaled)
        rawMatrix = rawSignal.reshape(windowNumber,channelNumber)
        MAF = MAFCalculation(rawMatrix, windowNumber, channelNumber, windowSize=3)

    
        yX = (np.array(MAF) >= 0).astype(int)
        y = predictedYPostprocessing(yX)
        print(y.shape)
        yPredictedAfterReshaping = collarTechnique(y, collar = 2)

        # predict
        #yPredictedBeforeReshaping = svc_clf.predict(XTestScaled)

        # reshape to (windows, channels)
        #Ymatrix = yPredictedBeforeReshaping.reshape(windowNumber, channelNumber)

        # postprocess
        #yPredictedAfterReshaping = predictedYPostprocessing(Ymatrix)


        # evaluation
        print("Patient: ", i)
        print("positives in test y:", CalculateNumberOfSeizures(YTestPerPatient))
        print("positives in predicted y:", CalculateNumberOfSeizures(yPredictedAfterReshaping))

        testAccurancy = accuracy_score(YTestPerPatient[:len(yPredictedAfterReshaping)], yPredictedAfterReshaping)
        print("Test accuracy of SVM:", round(testAccurancy,2))

        cmTest = confusion_matrix(YTestPerPatient[:len(yPredictedAfterReshaping)], yPredictedAfterReshaping, labels=[0, 1])
        print("Test confusion matrix:\n", cmTest)

         
        TN = cmTest[0,0]
        FP = cmTest[0,1]
        FN = cmTest[1,0]
        TP = cmTest[1,1]

        sumFN = sumFN + FN
        sumTN = sumTN + TN
        sumFP = sumFP + FP
        sumTP = sumTP + TP

        sensitivity, specificity, accurancy, precision, fpr = calculateMetrics(TP,FP,FN,TN)
        print("Test accuracy of SVM:", round(accurancy, 2))
        if sensitivity is not None:
            print("Test sensitivity of SVM:", round(sensitivity, 2))
        else:
            print("Test sensitivity is none")
        print("Test specificity of SVM:", round(specificity, 2))

        if precision is not None:
             print("Test precision of SVM:", round(precision, 2))
        else:
            print("Test precision is none")

        print("Test fpr of SVM:", round(fpr, 2))

        sens.append(sensitivity)
        acc.append(accurancy)
        spec.append(specificity)
        prec.append(precision)
        fpriii.append(fpr)

    averageAccurancy = np.mean(acc)
    averagesensitivity = np.nanmean(sens)
    averagespecificity = np.mean(spec)
    avearageprecision = np.nanmean(prec)
    averagefpr = np.mean(fpriii)
    print("Test average accuracy of SVM:", round(averageAccurancy,2))
    print("Test average sensitivity of SVM:", round(averagesensitivity,2))
    print("Test average specificity of SVM:", round(averagespecificity,2))
    print("Test average precision of SVM:", round(avearageprecision,2))
    print("Test average fpr of SVM:", round(averagefpr,2))

    sumedConfusionMatrix = np.matrix([[sumTN, sumFP],
                  [sumFN, sumTP]])
    print("summed confusion matrix: ", sumedConfusionMatrix)
    

         

def collectSeizureNumberFromTxt(index,lines):

      #collect seizure number
        index = index + 1
        seizureNumber = lines[index].split(":")[1]
        seizureNumberConvertedToInt = int(seizureNumber)
        return seizureNumberConvertedToInt

def calculateSeizureIntervals(seizureNumberConvertedToInt,index, lines):
    seizure_intervals=[]
    zSTart = 1
    zEnd = 2
    index = index + 1
    for i in range(seizureNumberConvertedToInt):
                    seizureStart = lines[index + (2*i + zSTart)].split(":")[1]
                    seizureStartFinal =  int(seizureStart.split()[0])
                    seizureEnd = lines[index  + (2*i + zEnd)].split(":")[1]
                    seizureEndFinal =  int(seizureEnd.split()[0])
                    seizure_intervals.append((seizureStartFinal, seizureEndFinal))
    return seizure_intervals
     


def formatingSeizureIntervals(txtFilesPaths, word):
    seizure_intervals = []
    print(word)

    # if txt file exists please read it
    if len(txtFilesPaths) > 0:  
        with open(txtFilesPaths[0]) as f:
            lines = f.readlines()
        index = -1
        # find edf name file with in txt file and get the index that we know where to start collecting info
        for row in lines:
            if row.find(word) != -1:
                print('word is in file')
                print('Line Number:', lines.index(row))
                index = lines.index(row)
               
        # if we didnt find edf file name we say no seizures happened in that file
        if index == -1:
            return seizure_intervals
           

        print("index: ",index)

        seizureNumberConvertedToInt = collectSeizureNumberFromTxt(index,lines)
        seizure_intervals = calculateSeizureIntervals(seizureNumberConvertedToInt, index, lines)
        return seizure_intervals
      

      #sita i catch perdaryt

    else:
        print("NO txt file in folder")

    return seizure_intervals

def concatinatingXYSeizuresAndXYNonseizures(XTrain,XRand,yTrain,YRand):

        XBALANCEDALL = XTrain + XRand
        YBALANCEDALL = yTrain + YRand

        XFINAL = []
        for element in XBALANCEDALL:
            if  element.size > 0:
             XFINAL.append(element)

        YFINAL = []
        for element in YBALANCEDALL:
            if  element.size > 0:
             YFINAL.append(element)

        XConcatinated = np.vstack(XFINAL)
        YConcatinated = np.hstack(YFINAL)

        print("X_all shape:", XConcatinated.shape)
        print("Y_all shape:", YConcatinated.shape)

        return XConcatinated, YConcatinated



#save model https://stackoverflow.com/questions/56107259/how-to-save-a-trained-model-by-scikit-learn
def Main(seconds):
        
        totalSeizuresWindows = 0
        totalNonSeizureWindows = 0

        edfFileNames = []

        yTest = []
        XTest = []
        XTrain = []
        yTrain = [] 
        XRand =[]
        YRand = []
        seizures = []
        patientsNumber = 0
        #read all files in folders, it is not filtered
        mypath = "/home/justina/Documents/EEG"
       
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        print(onlyfiles)

        #txt files in folder
        txtFilesPaths = glob.glob(join(mypath, "*.txt"))

        # print all txt files paths and get txt file names
        print("TXT files:", txtFilesPaths)
        txtFileNames = []
        for f in txtFilesPaths:
            txtFileNames.append(ntpath.basename(f))

        print(txtFileNames)

        #edf files in folder     
        # get all edf files paths in folder
        edfFilesPaths = glob.glob(join(mypath, "*.edf"))
        print("EDF:", edfFilesPaths)
        for f in edfFilesPaths:
            patientsNumber = patientsNumber + 1
            # get edf file names in given folder
            edfFileNames.append(ntpath.basename(f))
            # read txt file and put seizure start and end times to tuple list
            seizure_intervals = formatingSeizureIntervals(txtFilesPaths, ntpath.basename(f))
            print("seizure_intervals: ", seizure_intervals)
            # read edf and get metadata
            raw = mne.io.read_raw_edf(f, preload=True, infer_types=True)
            info = raw.info 
            raw.set_meas_date(None)
            #raw.plot()
            print(info)
            #frequancy
            frequency = raw.info["sfreq"]
            print(frequency)
            #channel names  
            channelNames = raw.ch_names   
            print("Channel names: ", channelNames)  
            #sample size  
            sampleSize = raw.n_times
            print("Sample size: ", sampleSize)
            # skaidome po 30 sek langus
            oneWindowSignalsNumber =  seconds * int(frequency)
            #kiek is viso langu po 30 sek bus
            windowNumber = int (sampleSize / oneWindowSignalsNumber)
            #kiek kanalu is viso
            channelNumber = len(raw.ch_names)
            print(oneWindowSignalsNumber)
            print(windowNumber)
            # preprocess signals
            signals = preprocessSignals(raw, notchFreq = 50, bandLowFreq = 0.5, bandHighFreq = 120)
            # split to windows
            window = splitToWindows(signals, channelNumber, windowNumber, oneWindowSignalsNumber)
            # calculate all features in all segments
            featureVectorALLSegments = featureCalculation(signals, window, frequency,channelNumber,seconds)
            print(featureVectorALLSegments.shape) #(2760=120*23,12)
            XTest.append(featureVectorALLSegments)
            # fill y vector with seizures
            y = fillYWithSeizures( windowNumber,seizure_intervals, seconds)
            yTest.append(y)
            print(y.shape)  #(120,)
            print(y)

            # calculate number of seizures
            seizuresNumber = CalculateNumberOfSeizures(y)
            seizures.append(seizuresNumber)
            print(seizuresNumber)

            # length of window
            print("============")
            print("window:",len(window))
            print("============")


            #GET X AND Y VALUES OF SEIZURES AND ALSO USING THIS SIGNALS OF X CALCULATE FEATURES            
            XSeizures, ySeizures = getXAndYAtSpecialIdexesWhenSeiziure(windowNumber, seizure_intervals, seconds,signals, channelNumber,oneWindowSignalsNumber)
            print("X segments of seizures: ", XSeizures)
            print("y segments of seizures: ", ySeizures)
            seizuresWindowsNumber = int(np.sum(y))
            totalSeizuresWindows = totalSeizuresWindows + seizuresWindowsNumber
            featureVectorSeizureSegments =  featureCalculation(signals, XSeizures, frequency,channelNumber,seconds)
            XTrain.append(featureVectorSeizureSegments)
            yTrain.append(np.repeat(ySeizures, channelNumber))
            print("============")
            print("X seizures:", len(XSeizures))
            print("============")


            # GET X AND Y VALUES OF NON SEIZURES AND ALSO USING THIS SIGNALS OF X CALCULATE FEATURES
            XRandomNonseizure, yRandomNonSeizure = getXAndYAtSpecialIdexesWhenNONSeiziure(windowNumber, seizure_intervals, seconds,signals, channelNumber,oneWindowSignalsNumber, seizuresWindowsNumber)
            featureVectorNonSeizureSegments =  featureCalculation(signals, XRandomNonseizure, frequency,channelNumber,seconds)
            print("X segments of non seizure: ",  XRandomNonseizure,)
            print("y segments of non seizure : ", yRandomNonSeizure)
            NonseizuresWindowsNumber = int(np.sum(yRandomNonSeizure==0))
            totalNonSeizureWindows = totalNonSeizureWindows + NonseizuresWindowsNumber
            XRand.append(featureVectorNonSeizureSegments)
            YRand.append(np.repeat(yRandomNonSeizure, channelNumber))
            print("============")
            print("X random seizures:", len(XRandomNonseizure))
            print("============")

    
        print("Y VECTOR: ", yTrain)
        print("features vectors: ", XTrain)
        print("total seizure windows: ", totalSeizuresWindows)
        print("total non seizure windows: ", totalNonSeizureWindows)

        XConcatinated, YConcatinated = concatinatingXYSeizuresAndXYNonseizures(XTrain,XRand,yTrain,YRand)
      
        calculateSVM(XTest, yTest, XConcatinated, YConcatinated,channelNumber,seizures, patientsNumber, windowNumber, train_nr=15)




Main(seconds=4)
            









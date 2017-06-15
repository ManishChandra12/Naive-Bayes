import csv
import random

def loadCsv(filename):
    lines = csv.reader(open(filename, 'r'))
    dataset = list(lines)
    for i in range(len(dataset)):
        for j in [1,2]:
            dataset[i][j] = float(dataset[i][j])
    return dataset

def roundOffDataset(dataset):
    for i in range(len(dataset)):
        dataset[i][1]=round(dataset[i][1])
        dataset[i][2]=round(dataset[i][2],-1)
    return dataset

def splitDataset(dataset,splitRatio):
    trainSize = int(len(dataset)*splitRatio)
    trainSet = []
    copy = dataset
    while len(trainSet)<trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]

def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    if('overweight' not in list(separated.keys())):
        separated['overweight']=[]
    elif('underweight' not in list(separated.keys())):
        separated['underweight']=[]
    elif('normal' not in list(separated.keys())):
        separated['normal']=[]

    return separated

def generateHash(dataset):
    probMale = (sum(1 for i in range(len(dataset)) if dataset[i][0]=='M')+1)/(len(dataset)+2)
    probFemale = (sum(1 for i in range(len(dataset)) if dataset[i][0]=='F')+1)/(len(dataset)+2)
    probSex = {'M': probMale, 'F': probFemale}

    probHeight = {}
    heights = [4,5,6,7]
    for h in heights:
        probHeight[h] = (sum(1 for i in range(len(dataset)) if dataset[i][1]==h)+1)/(len(dataset)+len(heights))

    probWeight = {}
    weights = [x*10+30 for x in range(8)]
    for w in weights:
        probWeight[w] = (sum(1 for i in range(len(dataset)) if dataset[i][2]==w)+1)/(len(dataset)+len(weights))

    classes = ['overweight', 'normal', 'underweight']
    hashTable = {}
    separated = separateByClass(dataset)
    for i in classes:
        for j in ['M','F']:
            hashTable[tuple([i,j])] = ((sum(1 for p in range(len(separated[i])) if separated[i][p][0]==j)+1)/(len(separated[i])+2))
        for k in heights:
            hashTable[tuple([i,k])] = ((sum(1 for p in range(len(separated[i])) if separated[i][p][0]==k)+1)/(len(separated[i])+len(heights)))
        for l in weights:
            hashTable[tuple([i,l])] = (sum(1 for p in range(len(separated[i])) if separated[i][p][2]==l)/(len(separated[i])+len(weights)))

    for s in classes:
        hashTable[s] = (sum(1 for i in range(len(dataset)) if dataset[i][3]==s)+1)/(len(dataset)+len(classes))

    return hashTable, probSex, probHeight, probWeight

def calculateClassProbabilities(hasht,sex,height,weight, inputVector):
    probabilities = {}
    for cls in ['overweight','normal','underweight']:
       probabilities[cls] = (hasht[cls]/(sex[inputVector[0]]*height[inputVector[1]]*weight[inputVector[2]]))*(hasht[(cls,inputVector[0])]*hasht[(cls,inputVector[1])]*hasht[(cls,inputVector[2])])
    return probabilities

def predict(hasht,sex,height,weight, inputVector):
    probabilities = calculateClassProbabilities(hasht,sex,height,weight, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in list(probabilities.items()):
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel

def getPredictions(hasht,sex,height,weight, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(hasht,sex,height,weight, testSet[i])
        predictions.append(result)
    return predictions

def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0

def main():
    filename = b'C:\Users\Manish\Desktop\dmAssignment1.csv'
    splitRatio = 0.67
    accuracy=[]
    for i in range(100):
        dataset = loadCsv(filename)
        cpp = roundOffDataset(dataset)
        trainingSet, testSet = splitDataset(cpp, splitRatio)
        hasht,sex,height,weight = generateHash(trainingSet)
        predictions = getPredictions(hasht,sex,height,weight, testSet)
        accuracy.append(getAccuracy(testSet, predictions))
    acc = sum(accuracy)/len(accuracy)
    print('The average accuracy is {0}%'.format(acc))

main()
import random
from methods.read_data import createMatrix
from models.clustering import standardize
from methods.normalize_data import coalesceData, standardizeData
from pathlib import Path
import csv
from evaluation.conf_matrix import evaluate
from evaluation.holdout_validation import holdout_validation, holdout_split

CLASS_VALUES = {
    "INCREASE":1,
    "RETAIN":0,
    "DECREASE":-1
}

def getDatasetDir():
    PROJECT_DIR = Path(__file__).resolve().parent.parent
    return PROJECT_DIR / "PROJECT_DATASET"

def getLabelsDir():
    PROJECT_DIR = Path(__file__).resolve().parent.parent
    return PROJECT_DIR / "PROJECT_DATASET" /"province_policy_results.csv"

"""
reads the true class label of every province from the class field in the csv file. INCREASE becomes 1, RETAIN becomes 0 and DECREASE becomes -1.
the labeled dataset (province_policy_results.csv) is created by methods/label_data.py
"""
def read_labels(labels_path):
    with open(labels_path, "r", encoding="utf-8-sig") as file:
        rows = list(csv.reader(file))

    header = rows[0]
    class_idx = None
    for j in range(len(header)):
        if header[j].lower() == "class":
            class_idx = j
    if class_idx is None:
        print("Labels not found.")
        exit(1)

    labels = {}
    for row in rows[1:]:
        if row:
            word = row[class_idx].strip().upper()
            labels[row[0].strip()] = CLASS_VALUES[word]
    return labels

"""
loads the dataset, cleans it and gets the true class of every province
 
steps:
1. createMatrix reads the dataset into a matrix, one row per province
2. missing values are replaced with 0 and the features are standardized
3. the class of every province is read from the labels file
 
args:
- data_set_dir = folder of the dataset csv files
- labels_path = path of the labels csv file
 
return:
- provinces = list of the province names
- featuresList = list of the standardized numerical values per field for each province
- labelList = list of the true class of each province (1, 0 or -1)
all three lists are in the same order, index i is the same province in all of them
"""
def prepareData(data_set_dir, labels_path):
    provinces, fields, matrix = createMatrix(Path(data_set_dir))

    # Coalesce NA values to 0 for clustering, then standardize it.
    cleaned_data = standardizeData(coalesceData(matrix))
    # Create dictionary of province statistics with province name as key and list of features as value
    province_stats = {provinces[i]: cleaned_data[i] for i in range(len(provinces))}

    scaled_data = standardize(province_stats)
    true_labels = read_labels(labels_path)

    featuresList = []
    labelList = []

    for province in provinces:
        featuresList.append(scaled_data[province])
        labelList.append(true_labels[province])

    return provinces, featuresList, labelList

"""
computes the Minkowski distance between two rows. p = 1 Manhattan distance and p = 2 Euclidean distance
 
args:
- test_row = list of the feature values of the row that will be classified
- training_row = list of the feature values of a training row without its class
- p = exponent of the distance
 
return:
- distance between the two rows
"""
def distance(test_row, training_row, p):
    total_distance = 0
    for i in range(len(test_row)):
        total_distance += abs(test_row[i]-training_row[i]) ** p
    return total_distance ** (1/p)

"""
finds the k training rows that are the nearest to test row
 
args:
- test_instance = feature values of one test row 
- trainingList = list of training rows, the features with the class as the last element
- k = number of neighbors
- p = exponent of the distance
 
return:
- list of the k nearest neighbors, nearest first. each neighbor is a tuple (class, distance, position of the row in trainingList)
"""
def knn(test_instance, trainingList, k, p):
    #one row of test to all training
    distances = []
    for row in range(len(trainingList)):
        #exclude the last column (the cluster from clustering) from the distance
        dist = distance(test_instance, trainingList[row][:-1], p)
        #append the cluster, the distance and the position of the training row
        distances.append((trainingList[row][-1], dist, row))

    distances.sort(key=lambda x: x[1])
    return distances[:k]

"""
gets the predicted class from the neighbors by majority. if two classes are tied, the class of the nearest neighbor is chosen
because the neighbors are sorted from nearest to farthest and max returns the first most common class
 
args:
- neighbors = list of tuples from knn, the class is the first element of each tuple
 
return:
- the most common class among the neighbors
"""
def classify(neighbors):
    classes = []
    for neighbor in neighbors:
        #append only the label
        classes.append(neighbor[0])

    return max(classes, key=classes.count)

"""
predicts the class of every test row using its k nearest training rows
 
args:
- testingList = list of test rows, no label only features
- trainingList = list of training rows, the features with the class as the last element
- k = number of neighbors
- p = exponent of the distance
 
return:
- new list of the test rows with the predicted class added as the last element of each row. testingList is not changed
"""
def outputMatrix(testingList, trainingList, k, p):
    #add the predicted cluster to each test row
    output = []
    for i in range(len(testingList)):
        neighbors = knn(testingList[i], trainingList, k, p)
        test_class = classify(neighbors)
        #new row, so the test row itself is not changed
        output.append(testingList[i] + [int(test_class)])
    
    return output

#same as in evaluation folder with specified model
def kFoldSplit(num_regions, folds):
    #shuffle the positions of the regions
    positions = list(range(num_regions))
    random.seed(1) #for reproducibility
    random.shuffle(positions)

    foldList = []
    for f in range(folds):
        foldList.append([])

    for i in range(num_regions):
        foldList[i % folds].append(positions[i])

    #returns a 2d array of shuffled integers for region idx
    return foldList
 
def crossValidation(featuresList, labelList, foldList, k, p):
    #the regions are split into folds, each fold is the test set once, other folds are training set
    folds = len(foldList)
 
    results = []
    for f in range(folds):
        trainingList = []   #data of same region with cluster label at end
        trainingIndex = []  #which region each training row is
        for g in range(folds):
            #the training set is all folds except the test fold
            if g != f:
                #iterate the 2d array of integers for region idx
                for i in foldList[g]:
                    trainingList.append(featuresList[i] + [labelList[i]])
                    trainingIndex.append(i)
 
        testingRows = []
        testingFold = foldList[f] #list of integers in the test fold    
        for i in testingFold:
            testingRows.append(featuresList[i])

        output = outputMatrix(testingRows, trainingList, k, p)

        for row_idx in range(len(testingRows)):
            predicted = output[row_idx][-1]

            #the regions that were the neighbors
            nearest = []
            for neighbor in knn(testingRows[row_idx], trainingList, k, p):
                #append the index of the region in the training set
                nearest.append(trainingIndex[neighbor[2]])

            #(region, real label, predicted label, neighbors, fold)
            i = testingFold[row_idx]
            results.append((i, labelList[i], predicted, nearest, f + 1))

    #back to the order of the regions
    results.sort(key=lambda x: x[0])
    return results
 
def accuracy(results):
    correct = 0
    for result in results:
        if result[1] == result[2]:
            #if the predicted label is correct
            correct += 1
    return correct / len(results)

def printOutput(k, p, regions, labelList, results):
    print("K-Nearest Neighbor Output from the provinces")
    print("k: " + str(k) + "\t" "p: " + str(p))
    print("class: 1 = INCREASE, 0 = RETAIN, -1 = DECREASE")
    print("\n")
    print(f"{'Province':<22}{'Fold':<6}{'True':<9}{'KNN':<6}{'':<3}Nearest provinces")
    for result in results:
        if result[1] == result[2]:
            mark = "/" #when the predicted cluster is correct
        else:
            mark = "x"
        nearest = ""
        for j in result[3]:
            nearest += regions[j] + " (" + str(labelList[j]) + "), "
        #print region, real cluster, predicted cluster, neighbors, fold
        print(f"{regions[result[0]]:<22}{result[4]:<6}{result[1]:<9}{result[2]:<6}{mark:<3}{nearest[:-2]}")

def printFolds(regions, foldList):
    for f in range(len(foldList)):
        names = []
        for i in foldList[f]:
            names.append(regions[i])
        print("Fold " + str(f + 1) + ": " + ", ".join(names))

def printAccuracy(results):
    correct = 0
    for result in results:
        if result[1] == result[2]:
            #if the predicted cluster is correct
            correct += 1
    print("\nAccuracy:", round(correct / len(results), 4))

def printConfMatrix(results):
    actual = []
    predicted = []
    for result in results:
        actual.append(result[1])
        predicted.append(result[2])
    return evaluate(actual, predicted)

def runHoldout(features_list, label_list,train_ratio, k, p):
    def model(test, train):
        return outputMatrix(test, train, k, p)
    return holdout_validation(features_list, label_list, train_ratio, model)

def printHoldout(featureList, labelList, regions, train_ratio, k, p):
    results = runHoldout(featureList, labelList, train_ratio, k, p)
    print("K-Nearest Neighbor Output from the provinces (holdout)")
    print("k: " + str(k) + "\t" "p: " + str(p) + "\t" "train ratio: " + str(train_ratio))
    print("class: 1 = INCREASE, 0 = RETAIN, -1 = DECREASE")
    print("only the test provinces are shown, the other provinces were used for training")
    print("\n")
    print(f"{'Province':<22}{'True':<9}{'KNN':<6}")
    for result in results:
        if result[1] == result[2]:
            mark = "/" #when the predicted class is correct
        else:
            mark = "x"
        print(f"{regions[result[0]]:<22}{result[1]:<9}{result[2]:<6}{mark}")

def main():
    #5 clusters for now
    folds, k, p, train_ratio = 5, 3, 2, 0.8

    data_set_dir = getDatasetDir()
    labels_dir = getLabelsDir()
    regions, featuresList, labelList = prepareData(data_set_dir, labels_dir)
    
    foldList = kFoldSplit(len(featuresList), folds)
    printFolds(regions, foldList)
    print()
    
    results = crossValidation(featuresList, labelList, foldList, k, p)
    printOutput(k, p, regions, labelList, results)
    printAccuracy(results)
    printConfMatrix(results)
    printHoldout(featuresList, labelList, regions, train_ratio, k, p)

main()


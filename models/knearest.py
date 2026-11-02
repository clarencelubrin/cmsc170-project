import random
from methods.read_data import createMatrix
from models.clustering import k_means_clustering, standardize
from pathlib import Path

def getDatasetDir():
    PROJECT_DIR = Path(__file__).resolve().parent.parent
    return PROJECT_DIR / "PROJECT_DATASET"

def prepareData(data_set_dir, clusters):
    regions, fields, matrix = createMatrix(Path(data_set_dir))
    region_stats = {}

    for i in range(len(regions)):
        row = []
        for value in matrix[i]:
            if value is None:
                row.append(0)
            else:
                row.append(value)
        region_stats[regions[i]] = row

    scaled_data = standardize(region_stats)
    categories = k_means_clustering(scaled_data, clusters)

    featuresList = []
    labelList = []

    for region in regions:
        featuresList.append(scaled_data[region])
        labelList.append(categories[region])

    return regions, featuresList, labelList


def distance(test_row, training_row, p):
    total_distance = 0
    for i in range(len(test_row)):
        total_distance += abs(test_row[i]-training_row[i]) ** p
    return total_distance ** (1/p)

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
 
def classify(neighbors):
    classes = []
    for neighbor in neighbors:
        #append only the cluster
        classes.append(neighbor[0])

    return max(classes, key=classes.count)
 
def outputMatrix(testingList, trainingList, k, p):
    #add the predicted cluster to each test row
    output = []
    for i in range(len(testingList)):
        neighbors = knn(testingList[i], trainingList, k, p)
        test_class = classify(neighbors)
        #new row, so the test row itself is not changed
        output.append(testingList[i] + [int(test_class)])
    
    return output

def kFoldSplit(num_regions, folds):
    #shuffle the positions of the regions
    positions = list(range(num_regions))
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

            #(region, real cluster, predicted cluster, neighbors, fold)
            i = testingFold[row_idx]
            results.append((i, labelList[i], predicted, nearest, f + 1))

    #back to the order of the regions
    results.sort(key=lambda x: x[0])
    return results
 
def accuracy(results):
    correct = 0
    for result in results:
        if result[1] == result[2]:
            #if the predicted cluster is correct
            correct += 1
    return correct / len(results)

def printOutput(k, p, regions, labelList, results):
    print("K-Nearest Neighbor Output from the regions")
    print("k: " + str(k) + "\t" "p: " + str(p))
    print("\n")
    print(f"{'Region':<13}{'Fold':<6}{'K-means':<9}{'KNN':<6}{'':<3}Nearest regions (cluster)")
    for result in results:
        if result[1] == result[2]:
            mark = "/" #when the predicted cluster is correct
        else:
            mark = "x"
        nearest = ""
        for j in result[3]:
            nearest += regions[j] + " (" + str(labelList[j]) + "), "
        #print region, real cluster, predicted cluster, neighbors, fold
        print(f"{regions[result[0]]:<13}{result[4]:<6}{result[1]:<9}{result[2]:<6}{mark:<3}{nearest[:-2]}")

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

def main():
    #5 clusters for now
    clusters, folds, k, p = 5, 5, 3, 2

    data_set_dir = getDatasetDir()
    regions, featuresList, labelList = prepareData(data_set_dir, clusters)
    
    foldList = kFoldSplit(len(featuresList), folds)
    printFolds(regions, foldList)
    print()
    
    results = crossValidation(featuresList, labelList, foldList, k, p)
    printOutput(k, p, regions, labelList, results)
    printAccuracy(results)

main()


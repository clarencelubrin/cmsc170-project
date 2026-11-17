import random
"""
splits the shuffled indices of the regions into specified fold

args: 
- num_regions = total amt of regions
- folds = specified amt of folds

return: 
- 2 dimensional array of integers ranging 0 to num_regions - 1
"""
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

"""
evaluates the model's output through the use of specified amt of folds. all folds will be the test case once and the rest are the training cases

args:
- featuresList = list of numerical values per fields for each row
- labelList = list of true label per region, same order as featuresList
- foldList = 2d array of region indices
- model = function model(testingRows, trainingList) which returns the testing rows with the predicted label added as the last element per row   

return:
- list of tuples (region index, true label, predicted label, fold number) sorted by region index. the fold number starts at 1
""" 
def crossValidation(featuresList, labelList, foldList, model):
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

        output = model(testingRows, trainingList)

        for row_idx in range(len(testingRows)):
            #(region, true label, predicted label, fold)
            i = testingFold[row_idx]
            results.append((i, labelList[i], output[row_idx][-1], f + 1))

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
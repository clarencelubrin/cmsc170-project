import random
def holdout_split(matrix, label_list, train_ratio):
    """
    holdout_split(matrix, label_list, train_ratio)
        - This function splits the data into test and training splits.
        - The train data will have labels, while the test data split will have its labels removed. 

    parameters:
        matrix      - This is the matrix after standardization & coalesce has been done to the original dataset matrix (N x M).
        label_list  - List of labels per row (has length N)
        train_ratio  - The train_ratio split in float e.g. 0.8 for 80% train (advisable to use 80/20 - train/test split because the regions/rows are limited.) 

    returns:
        training_list - rows of features with the label as the last element
        testing_rows  - rows of features only (no labels)
        test_labels   - true labels of the test rows, in the same order
        test_indices  - original row positions of the test rows
    """

    num_regions = len(matrix)
    if num_regions != len(label_list):
        raise ValueError("regions and label list length mismatch.")
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    positions = list(range(num_regions))
    random.seed(1) #for reproducibility
    random.shuffle(positions)

    train_size = round(num_regions * train_ratio) # num of regions * train_ratio, round make sures that it is an integer
    train_size = max(1, min(train_size, num_regions-1)) # this makes sure that the values are between 1-max-1. This means each split must have atleast 1 dataset each.

    train_indices = positions[:train_size] # from start to train_size
    test_indices = positions[train_size:] # from train_size to end

    # Train data splits retain their labels, so append it at the last column.
    training_list = []
    for i in train_indices:
        training_list.append(list(matrix[i]) + [label_list[i]])

    # Test data splits are features only, keep the labels in a different array.
    testing_rows = []
    test_labels = []
    for i in test_indices:
        testing_rows.append(list(matrix[i]))
        test_labels.append(label_list[i])

    return training_list, testing_rows, test_labels, test_indices

def holdout_validation(matrix, label_list, train_ratio, model):
    """
    holdout_validation(matrix, label_list, train_ratio, model)
        - Gets split data, then runs a specified model with those split data.
        - Stores result in an array containing a tuple (region_idx, true label, predicted label, fold)

    parameters:
        matrix       - This is the matrix after standardization & coalesce has been done to the original dataset matrix (N x M).
        label_list   - List of labels per row (has length N)

        train_ratio  - The train_ratio split in float e.g. 0.8 for 80% train 
                       (advisable to use 80/20 - train/test split because the regions/rows are limited.) 

        model        - A function model(testing_rows, training_list) returning rows with the predicted label last. 
                       Wrap models that need extra parameters, 
                       e.g. lambda test, train: outputMatrix(test, train, 3, 2)

    returns:
        results      - A list containing a tuple (region_idx, true label, predicted label, fold). 
                       Keep in mind that fold is used here so that we can just reuse the accuracy function 
                       defined in /evaluation/kfold_cross_validation.py
    """
    # Gets the split data
    training_list, testing_rows, test_labels, test_indices = holdout_split(matrix, label_list, train_ratio)

    # Runs the model using the split data
    output = model(testing_rows, training_list)

    # Stores the output in results[] as a tuple (region index, true label, predicted label, fold)
    results = []
    for j in range(len(testing_rows)):
        # (region index, true label, predicted label, fold)
        results.append((test_indices[j], test_labels[j], output[j][-1], 1))

    results.sort(key=lambda x: x[0])
    return results
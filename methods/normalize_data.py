"""
The following functions (normalizeData, standardizeData) transforms the data into a standard range. 
Since the datasets used different scales for their measurement, this step is needed in order to ensure equal contribution of features, and improve model performance.

normalizeData: Good for models that rely on distance metrics, by preventing features with larger scales from dominating the learning process.
standardizeData: Since values are not in a fixed range like in min-max scaling, training converges faster for MLP because weight updates aren't in the same direction everytime. And it also handles outliers better because instead of relying on only the Max and Min values, it relies on the average distance of all points from the mean. 
References: https://www.geeksforgeeks.org/machine-learning/what-is-data-normalization/, https://www.geeksforgeeks.org/machine-learning/standardization-in-machine-learning/, https://www.datacamp.com/tutorial/normalization-in-machine-learning
"""
from numpy import mean, std 

def normalizeData(data):
    regions = len(data)
    features = len(data[0])
    normalizedData = [[0]*features for _ in data]

    for i in range(features):
        column = [data[r][i] for r in range(regions)] # Coalesce data to 0 if None
        x_min, x_max = min(column), max(column)
        for j in range(regions): 
            x = data[j][i] 
            if x_max == x_min:
                normalizedData[j][i] = 0.0
            else: 
                normalizedData[j][i] = float((x - x_min) / (x_max - x_min)) # Min-Max Scaling Formula
    return normalizedData


def standardizeData(data):
    regions = len(data)
    features = len(data[0])
    normalizedData = [[0]*features for _ in data]

    for i in range(features):
        column = [0 if data[r][i] == None else data[r][i] for r in range(regions)] # Coalesce data to 0 if None
        x_mean, x_std = mean(column), std(column)
        for j in range(regions): 
            x = 0 if data[j][i] == None else data[j][i] 
            if x_std == 0:
                normalizedData[j][i] = 0.0
            else:
                normalizedData[j][i] = float((x - x_mean) / x_std) # Z-score Scaling Formula
    return normalizedData

"""
The following functions (coalesceData, imputateData, dropFeature) deals with missing data. 

References: https://www.datacamp.com/tutorial/techniques-to-handle-missing-data-values, 
"""
def coalesceData(data):
    return [[0 if feature is None else feature for feature in region] for region in data]

# def imputateData(data): implement one or the other after presenting 
# def dropFeature(data):
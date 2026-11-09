"""
Confusion matrix and performance metrics implementation.
"""

def confusion_matrix(actual: list, predicted: list, positive) -> dict:
  """
  Calculates the binary confusion matrix of a classifier
  Args:
  - actual: List of the actual (true) labels
  - predicted: List of the predicted labels
  - positive: The label that is treated as the positive class
  Returns:
  - matrix: A dictionary with the counts of TP, FN, FP and TN
  """
  if len(actual) != len(predicted):       # Both lists must describe the same samples
    raise ValueError("actual and predicted must have the same length")

  matrix = {"TP": 0, "FN": 0, "FP": 0, "TN": 0}   # Initialize the counts

  # Loop through all samples and compare the actual and predicted label
  for i in range(len(actual)):
    is_actual_positive = actual[i] == positive
    is_predicted_positive = predicted[i] == positive
    if is_actual_positive and is_predicted_positive:        # Correctly identified as positive
      matrix["TP"] += 1
    elif is_actual_positive and not is_predicted_positive:  # Positive but predicted as negative
      matrix["FN"] += 1
    elif not is_actual_positive and is_predicted_positive:  # Negative but predicted as positive
      matrix["FP"] += 1
    else:                                                   # Correctly identified as negative
      matrix["TN"] += 1
  return matrix

def recall(matrix: dict) -> float:
  """
  The proportion of actual positive cases that are correctly identified as positive.
  Recall = TP / (TP + FN)
  """
  denominator = matrix["TP"] + matrix["FN"]
  # If there are no actual positives, the recall is undefined, so just use 0
  return matrix["TP"] / denominator if denominator != 0 else 0

def precision(matrix: dict) -> float:
  """
  The proportion of positive predictions that were actually correct.
  Precision = TP / (TP + FP)
  """
  denominator = matrix["TP"] + matrix["FP"]
  # If there are no positive predictions, the precision is undefined, so just use 0
  return matrix["TP"] / denominator if denominator != 0 else 0

def f1_score(matrix: dict) -> float:
  """
  A harmonic mean of precision and recall, providing a balance between the two.
  F1 Score = (2 * Recall * Precision) / (Recall + Precision)
  """
  r = recall(matrix)
  p = precision(matrix)
  # If both recall and precision are 0, the F1 score is undefined, so just use 0
  return (2 * r * p) / (r + p) if (r + p) != 0 else 0

def accuracy(matrix: dict) -> float:
  """
  The overall proportion of correct predictions (both positive and negative) out of the total number of cases.
  Accuracy = (TP + TN) / Total no. of samples
  """
  total = matrix["TP"] + matrix["FN"] + matrix["FP"] + matrix["TN"]
  # If there are no samples, the accuracy is undefined, so just use 0
  return (matrix["TP"] + matrix["TN"]) / total if total != 0 else 0

def print_confusion_matrix(matrix: dict, positive="Positive"):
  """
  Print the confusion matrix in the layout of
                Predicted
                Positive  Negative
  Actual Positive  TP        FN
         Negative  FP        TN
  """
  width = max(len(str(value)) for value in matrix.values())   # Get the width of the largest count
  width = max(width, len("Positive"))                         # The column must fit the header
  print(f"Confusion Matrix (Positive class: {positive})")
  print(f"{'':18}{'Predicted':^{width * 2 + 1}}")
  print(f"{'':18}{'Positive':>{width}} {'Negative':>{width}}")
  print(f"{'Actual':8}{'Positive':>10}{matrix['TP']:>{width}} {matrix['FN']:>{width}}")
  print(f"{'':8}{'Negative':>10}{matrix['FP']:>{width}} {matrix['TN']:>{width}}")

def evaluate(actual: list, predicted: list, positive=None) -> dict:
  """
  Calculates the confusion matrix and its performance metrics
  Args:
  - actual: List of the actual (true) labels
  - predicted: List of the predicted labels
  - positive: The label that is treated as the positive class (optional)
              If not given, each label is evaluated as the positive class (one vs. rest).
  Returns:
  - results: A dictionary mapping each positive label to its confusion matrix and metrics.
  """
  if positive is None:          # Get all the labels found in both lists
    labels = sorted(set(actual) | set(predicted), key=str)
  else:
    labels = [positive]

  results = {}    # Create the results
  for label in labels:
    matrix = confusion_matrix(actual, predicted, label)   # Calculate the confusion matrix
    results[label] = {
      "matrix": matrix,
      "recall": recall(matrix),
      "precision": precision(matrix),
      "f1_score": f1_score(matrix),
      "accuracy": accuracy(matrix),
    }

    # Print and display the confusion matrix and the metrics
    print_confusion_matrix(matrix, label)
    print(f"Recall:    {results[label]['recall']:.4f}")
    print(f"Precision: {results[label]['precision']:.4f}")
    print(f"F1 Score:  {results[label]['f1_score']:.4f}")
    print(f"Accuracy:  {results[label]['accuracy']:.4f}\n")
  return results

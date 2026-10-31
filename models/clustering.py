"""
K-means clustering algorithm implementation.
"""
import math
import random

def distance(X: list, V: list, p=2) -> float:
  """
  Calculate the distance of vector X (test) and feature vector V with len(X) dimensions
  Euclidean distance with p=2.
  """
  value = 0
  # Loop through all dimensions of the vector
  for i in range(len(X)):
    value += pow(abs(X[i] - V[i]), p)
  return pow(value, 1/p)

def standardize(region_stats: dict) -> dict:
  """
  Scale each feature to mean 0 and standard deviation 1
  """
  ids = list(region_stats.keys())
  total_dimensions = len(region_stats[ids[0]])
  n = len(ids)
  scaled = {id: [] for id in ids}
  for dim in range(total_dimensions):
    mean = sum(region_stats[id][dim] for id in ids) / n
    std = math.sqrt(sum((region_stats[id][dim] - mean) ** 2 for id in ids) / n)
    for id in ids:
      # If a feature is constant, std is 0, so just use 0
      scaled[id].append((region_stats[id][dim] - mean) / std if std != 0 else 0)
  return scaled

def k_means_clustering(region_stats: dict, k: int, centroids=[]) -> dict:
  """
  Calculates for the K-Means clustering
  Args:
  - region_stats: Dictionary of region statistics with region id as key and list of features as value
  - k: Number of clusters
  - centroids: List of initial centroids (optional)
  Returns:
  - categories: A dictionary mapping each region to its assigned cluster.
  """
  e = 0                       # Minimum error for detecting if it still is converging

  if len(centroids) == 0:     # Get a random value of centroids
    centroids = random.sample(list(region_stats.values()), k)

  print("Initial Centroids")
  initial_centroids = [c.copy() for c in centroids] # Initialize the initial centroids
  for centroid in initial_centroids:                # Print and display the centroids
    print(tuple(centroid))

  categories = {}   # Create the categories
  iterations = 0    # Initialize the number of iterations

  while True:
    iterations += 1
    prev_centroids = [c.copy() for c in centroids]  # copy each vector, not just the outer list

    # Calculate what categories are the points
    for id, points in region_stats.items():
      min_distance = math.inf
      min_index = -1
      for centroid_idx in range(len(centroids)):
        centroid = prev_centroids[centroid_idx]     # Get the current centroid
        dist = distance(points, centroid)           # Calculate the euclidean distance
        if dist < min_distance:                     # Update if the distance is minimum
          min_distance = dist
          min_index = centroid_idx
      categories[id] = min_index                    # Update the categories dict

    # Update the centroids
    for centroid_idx in range(len(centroids)):
      total_dimensions = len(region_stats[list(region_stats.keys())[0]])  # Get the number of dimensions
      count = 0     # Count of datapoint
      for dim in range(total_dimensions):           # reset the centroid so the old value isn't added into the sum
        centroids[centroid_idx][dim] = 0
      for id, points in region_stats.items():
        if categories[id] == centroid_idx:    # If the category is for the centroid
          # increment the total for each dimension
          for dim in range(total_dimensions):
            centroids[centroid_idx][dim] += points[dim]
          count += 1                          # Add total count
      if count == 0:                          # Don't update if count is zero
        centroids[centroid_idx] = prev_centroids[centroid_idx].copy()  # restore the old centroid since it was reset above
        continue
      # Update the centroid
      for dim in range(total_dimensions):
        centroids[centroid_idx][dim] /= count

    # Check if ALL centroids have converged
    converged = True
    for centroid_idx in range(len(centroids)):
      if distance(prev_centroids[centroid_idx], centroids[centroid_idx]) > e:
        converged = False
        break

    # If converged, break the loop
    if converged:
      break

  # Print the final centroids and iterations
  print(f"\nFinal Centroids")
  for centroid in centroids:
    print(centroid)
  print(f"\nIterations: {iterations}")
  return categories

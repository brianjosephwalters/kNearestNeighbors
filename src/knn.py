'''
Created on Sep 18, 2014

@author: Brian J. Walters
'''

from parser import parseFile
from math import pow, sqrt
import heapq

"""
     Returns a normalized version of a 2D array using
     the provided basis list.
"""
def normalize(basis, matrix):
    if len(basis) != len(matrix[0]):
        print "normalize() ERROR: matrix rows must be the same length as the basis list!"

    normal = []
    for row in range(len(matrix)):
        normal.append([])
        for column in range(len(matrix[row])):
            normal[row].append(matrix[row][column] / basis[column])
    return normal

"""
    Returns a list with the maximum value for 
    each column of a 2D row-column array.
"""
def maxColumns(matrix):
    maximums = []
    for row in range(len(matrix)):
        for column in range(len(matrix[row])):
            if row == 0:
                maximums.append(matrix[row][column])
            elif matrix[row][column] > maximums[column]:
                maximums[column] = matrix[row][column]
    return maximums

"""
    Returns a list containing the euclidean distance between
    the provided row and all of the rows in the matrix.  
    NOTE: The returned list is indexed to the rows in the matrix.
"""
def rowDistances(row, matrix):
    if len(row) != len(matrix[0]):
        print "rowDistances() ERROR: unequal number of attributes."
    distances = []
    for currentRow in matrix:
        distances.append(rowDistance(row, currentRow))
    return distances

"""
    Returns the euclidean distance between two rows.
"""
def rowDistance(row1, row2):
    if len(row1) != len(row2):
        print "rowDistance() ERROR: unequal number of attributes."
    total = 0.0;
    for column in range(len(row1)):
        total += pow((row2[column] - row1[column]), 2)
    return sqrt(total)

"""
    Returns a list of K row-indexes for the K nearest neighbors
    to a given row.
    NOTE: row is the full row, not an index.
    NOTE: if k == 1, it would only return the _first_ matching row index 
          (which may be incorrect, since the corresponding outputs may be different).
    @requires: k > 1  
"""
def kNearestNeighbors(k, row, matrix):
    if k >= len(matrix):
        print "kNearestNeighbors() WARNING: k encompasses all of the data points!"
    distances = rowDistances(row, matrix)
    #heap = heapq.heapify(distances)
    min_values = heapq.nsmallest(k, distances)
    mins = []
    for value in min_values:
        mins.append(distances.index(value))
    return mins

"""
    Returns the average of the output values for 
    the indexes provided.
"""
def averageNeighborOutput(neighbors, outputMatrix):
    if len(outputMatrix[0]) != 1:
        print "averageNeighborOutput() ERROR: outputMatrix should have exactly 1 column."
    total = 0.0;
    for row_index in neighbors:
        total += outputMatrix[row_index][0]
    return total / len(neighbors)

"""
    Returns the error for a row based on the training
    data's output and the K nearest neighbors.
    NOTE: might be simpler not to use this method.
"""
def rowError(row_value, neighbors, outputMatrix):
    row_error = row_value - averageNeighborOutput(neighbors, outputMatrix)
    return row_error

def run(k):
    # 1) Load data files
    x_training = parseFile("../data/X_original.txt")
    y_training = parseFile("../data/Y_original.txt")
    x_testing = parseFile("../data/X_test.txt")
    y_testing = parseFile("../data/Y_test.txt")
    
    # 2) Get the maximums from the training data 
    #    (for normalization of _all_ data)
    x_maximums = maxColumns(x_training)
    y_maximums = maxColumns(y_training)
    
    # 3) Normalize data
    x_training_normal = normalize(x_maximums, x_training)
    y_training_normal = normalize(y_maximums, y_training)
    x_testing_normal = normalize(x_maximums, x_testing)
    y_testing_normal = normalize(y_maximums, y_testing)

    # 4) Compute error for each row in the Training Data.
    training_errors = []
    for row_index in range(len(x_training_normal)):
        # A) Get K nearest neighbor indexes for the row.
        #    When k = 1, we should just return the corresponding row index,
        #    since the error will be 0.  We only do this for the training data, though.
        if k == 1:
            neighbors = [row_index]
        else:
            neighbors = kNearestNeighbors(k, x_training_normal[row_index], x_training_normal)
        
        # B) Record the absolute value of the error.
        training_errors.append( abs(y_training_normal[row_index][0] - averageNeighborOutput(neighbors, y_training_normal)) )
        
    # 5) Get the MAE for the Training Data.
    training_mae = sum(training_errors)/len(training_errors)
    
    # 6) Get the R_MSE for the Training Data.
    training_r_mse = sqrt(sum( [pow(error,2) for error in training_errors] )/len(training_errors))
    
    # 7) Compute the error for each row of the Test Data.
    testing_errors = []
    for row_index in range(len(x_testing_normal)):
        # A) Get K nearest neighbor indexes for the row.
        neighbors = kNearestNeighbors(k, x_testing_normal[row_index], x_training_normal)
        
        # B) Record the absolute value of the error.
        testing_errors.append( abs( y_testing_normal[row_index][0] - averageNeighborOutput(neighbors, y_training_normal)) )
    
    # 8) Get the MAE for the Testing Data.
    testing_mae = sum(testing_errors)/len(testing_errors)
    
    # 9) Get the R_MSE for the Testing Data.
    testing_r_mse = sqrt(sum( [pow(error, 2) for error in testing_errors] )/len(testing_errors))
    
    return (training_mae, training_r_mse, testing_mae, testing_r_mse)

if __name__ == '__main__':
    f1 = open("../data/knn_mae.txt", 'w')
    f2 = open("../data/knn_r_mse.txt", 'w')
    
    for k in [1, 3, 5, 7, 9]:
        results = run(k)
        f1.write("K=" + str(k) + "," + str(results[0]) + ", " + str(results[2]) + "\n")
        f2.write("K=" + str(k) + "," + str(results[1]) + ", " + str(results[3]) + "\n")
    f1.flush()
    f2.flush()
    f1.close()
    f2.close()
    
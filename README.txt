Programming assignment for CSCI-6635: Pattern Recognition
Brian J. Walters

Implements the K-Nearest Neighbors algorithm.
Extends an original project (primarily spreadsheet work) which
calculated mean-absolute error (MAE), mean-squared error (MSE), 
and root mean-squared error (R_MSE) for training and test 
data that used linear model prediction:
Beta = inv(transform(X)*X)*tranform(X)*Y
and the regularized formula:
Beta = inv(transform(X)*X + lambda*I))*transform(X)*Y
where I is the identity matrix with I(1,1)=0.

This portion of the project uses the k-Nearest Neighbors
algorithm for model prediction, where k = 1, 3, 5, 7, and 9.
The results were added to the spreadsheet to compute error.

Implemented in python.
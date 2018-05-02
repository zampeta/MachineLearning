function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
C_values = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
minError = intmax;
for i=1:length(C_values)
  for j=1:length(C_values)
    C_potential = C_values(i);
    sigma_potential = C_values(j);
    % Calculate model/prediction/error for these C and sigma values.
    model = svmTrain(X, y, C_potential, @(x1, x2) gaussianKernel(x1, x2, sigma_potential));
    predictions = svmPredict(model, Xval);
    prediction_Error = mean(double(predictions ~= yval));
    % Calculated error is less than our previous best.
		% Keep track of the error and C/sigma values used so we can
		% compare against future predictions.
    
		if (prediction_Error <= minError)
			minError = prediction_Error;
			C = C_potential;
			sigma = sigma_potential;
    end
  end
end


% =========================================================================

end

function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% Add ones to the X data matrix
X = [ones(m, 1) X];
% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

% Unregularized Cost function
y_matrix = eye(num_labels)(y,:);
for i = 1:m
  X_1_layer = sigmoid(Theta1*X(i,:)');
  X_1_layer = [ones(1, 1); X_1_layer];
  J_i(i) =  (-y_matrix(i,:) *log(sigmoid(Theta2*X_1_layer))) - (1-y_matrix(i,:))*log(1-sigmoid(Theta2*X_1_layer)) ;
end
J = (1./m) *sum(J_i);
% Setup some useful variables
j_theta1_max = size(Theta1, 1);
k_theta1_max = size(Theta1, 2);

j_theta2_max = size(Theta2, 1);
k_theta2_max = size(Theta2, 2);

J_theta1_term = 0;
%Regularized Cost
for j = 1:j_theta1_max
  for k = 2:k_theta1_max
    J_theta1_term = J_theta1_term + (Theta1(j,k))**2 ;
  end
end
J_theta2_term =0;
for j = 1:j_theta2_max
  for k = 2:k_theta2_max
    J_theta2_term = J_theta2_term + (Theta2(j,k))**2 ;
  end
end
J_extra_term = (lambda/(2.*m))*(J_theta1_term+J_theta2_term);
J = J + J_extra_term;
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
D1 = zeros(size(Theta1));
D2 = zeros(size(Theta2));
 
% Part 2 - back propagation
for t = 1:m
 
 % Step 1: perform forward propagation
 
 a1 = [X(t,:)]; % 1 x 401
 
 z2 = a1 * Theta1'; % 1x25
 a2 = [1 sigmoid(z2)]; % (1 x 25) -&amp;gt; 1 x 26 
 
 z3 = a2 * Theta2'; % (1 x 26) * (26 x 10) = 1 by 10
 a3 = sigmoid(z3) ;% 1x10
 
 % Step 2: using y to calculate delta_L
 
 a = 1:num_labels; % a is a temp para
 Y = (a == y(t)); % making Y matrix as classification label
 
 d3 = a3 - Y; % 1 by 10
 
 % Step 3: backward propagation to calculate delta_L-1,
% delta_L-2, ... until delta_2. (this example only have one layer,
% so only need to calculate delta_2)
 
 d2 = Theta2' * d3'; % 26 x 1
 d2 = d2(2:end); % 25 x 1
 d2 = d2 .* sigmoidGradient(z2)';
 
 % Alternatively:
 %d2 = Theta2' * d3' .* a2' .* (1-a2)'; % 26 x 1
 %d2 = d2(2:end); % 25 x 1
 
 % Step 4: accumulate Delta value for all m input data sample
 % Theta1 has size 25 x 401
 % Theta2 has size 10 x 26
 
 D2 = D2 + d3' * a2; % 10 x 26
 D1 = D1 + d2 * a1; % 25 x 401
 
end
 


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Finally, calculate the gradient for all theta
Theta1_grad = 1/m*D1 + lambda/m*[zeros(size(Theta1,1),1) Theta1(:, 2:end)];
Theta2_grad = 1/m*D2 + lambda/m*[zeros(size(Theta2,1),1) Theta2(:, 2:end)];


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

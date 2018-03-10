function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    diff = theta' * X'; %dimensions
    hypo = (diff-y'); %dimensions
    I1 = sum(X'(1, :) .* hypo);
    I2 = sum(X'(2, :) .* hypo);

        %I1
        %I2

    temp1 = theta'(1,1) - (alpha * I1)/m;
    temp2 = theta'(1,2) - (alpha * I2)/m;
    %printf("\n\nBefore\nIteration: %d -> temp1: %f\n and theta: ", iter,temp1)
    %disp(theta)
    theta = [temp1; temp2];
    %printf("\n\nAfter\nIteration: %d -> temp2: %f\n and theta: ", iter,temp2)
    %disp(theta)





    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end

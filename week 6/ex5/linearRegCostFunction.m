function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%size(theta)
%size(X)
%X = [ones(m,1) X];

h = X * theta;
%h = theta' * X';

%size(h)
diff = (h-y).^2;
unreg = sum(diff)/(2*m);

%theta1 = zeros(size(theta));
theta1 = sum(theta(2:end,:).^2);
reg = (lambda*theta1)/(2*m);
J = unreg + reg;

grad1 = (X'*(h-y))/m;

%grad1 = sum((h-y).*X(:,2))/m;

%reg_grad = (theta*lambda)/m;
%reg_grad(1) = 0;
%grad = [grad1 + reg_grad];

reg_grad = zeros(size(grad));
reg_grad(2:end,:) = theta(2:end,:).*lambda/m;
grad = grad1 + reg_grad;


% =========================================================================

grad = grad(:);

end

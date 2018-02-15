function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


z = X*theta;
st1 = log(sigmoid(z));
st2 = y.*st1;
ky = 1.-y;
st3 = ky.*log(1-sigmoid(z));
st4 = st2 + st3;
st5 = -(sum(st4))/m;
temp1 = [0; theta(2:end,1:end).^2];
st6 = (lambda .* sum(temp1))/(2*m);

J = st6+st5;
theta1 = [0 ; theta(2:size(theta), :)];
grad = (X'*(sigmoid(z)-y) + lambda .* theta1)/m; %Don't know why summation won't occur here.
%part2 = lambda.*theta1;
%part1 = sum((sigmoid(z)-y));
%part3 = part1.*X;
%part4 = (part3 .+ part2)/m;
%grad = part4;
%grad = (sum((sigmoid(z)-y).*X) + lambda .* theta1)/m;


% =============================================================

end

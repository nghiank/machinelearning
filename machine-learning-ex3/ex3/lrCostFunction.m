function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 

S = sigmoid(X*theta);
T = log(S);
K = log(1-S);

grad = (1/m)* X' * (S - y);
n = size(theta,1);
J = (1/m)*sum( (-y).*T - (1-y).*K) + (lambda/(2*m))*sum(theta(2:n).^2);

tempG = (lambda/m) * theta;
tempG(1) = 0;
grad = grad + tempG;

% =============================================================


end

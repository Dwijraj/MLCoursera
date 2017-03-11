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


diff=X*theta-y;
Grad=diff;
diff=diff.^2;
cost=sum(diff);
reg=theta'*theta -theta(1)*theta(1);
reg=lambda*reg;
Net=(cost+reg);
divider=2*m;
J=Net/divider;

grad(1)=sum(X(:,1)'*Grad)/m;

n=size(X,2);

for i=2:n
    grad(i)=sum(X(:,i)'*Grad)/m  + (lambda/m)*theta(i) ;
end








% =========================================================================

grad = grad(:);

end

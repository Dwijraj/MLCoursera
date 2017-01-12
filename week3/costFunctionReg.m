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

Prob=log(sigmoid(X*theta));
ProbComp= log(ones(length(Prob),1)-sigmoid(X*theta));
cost=0;
for i=1:m
   
    if y(i)==1
     
        cost=cost+Prob(i);
    else
        cost=cost+ProbComp(i);

    end    
end
theta1=theta.^2;
theta1=theta1(2:length(theta1),1)
Value=ones(1,length(theta1))*theta1
RegularizationValue=(lambda*(Value/2))/m
J=-(cost/m)+RegularizationValue;
X_new=sigmoid(X*theta);
F=X'*(X_new-y);
Grad=F./m;%+(theta*(lambda/m));
Gradf=F./m+(theta*(lambda/m));
Gradf(1)=Grad(1);
grad=Gradf;



% =============================================================

end

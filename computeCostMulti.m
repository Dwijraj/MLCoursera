function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.



difference=(X*(theta))-y;
MeanSquare=difference.^2;
MeanSquare=MeanSquare(:,1);
m=length(MeanSquare);
sum=0;
i=1;

    while i<=m,
        sum=sum+MeanSquare(i);
        i=i+1;
    end     

format short;
J=0.5*(1/m)*sum;


% =========================================================================

end

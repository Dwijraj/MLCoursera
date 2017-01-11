function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
features=size(X,2);
mu = zeros(1, features);
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       
    i=1;
    while i<=features 
        
        Data=X(:,i);
        S=mean(Data);
        mu(1,i)=S;
        m=length(Data);
        Sub=ones(m,1);
        Sub=Sub.*S;
        Modified=Data-Sub;
        StandardDeviation=std(Data);
        sigma(1,i)=StandardDeviation;
        Modified=Modified./StandardDeviation;
        X_norm(:,i)=Modified;
        i=i+1;
    end     






% ============================================================

end

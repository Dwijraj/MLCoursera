function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions
                tp_count=0;
                fp_count=0;
                fn_count=0;
                predictions=(pval< epsilon);
                
                for i=1:length(predictions)
                   
                    if ((yval(i)==1) && (predictions(i)==1))
                        tp_count=tp_count+1;
                    elseif ((yval(i)==0) && (predictions(i)==1))
                         fp_count=fp_count+1;
                    
                    elseif ((yval(i)==1) && (predictions(i)==0))
                         fn_count=fn_count+1;
                       
                   end        
                end
                prec=(tp_count/(tp_count+fp_count));
                rec=(tp_count/(tp_count+fn_count));
   
                F1=((2*prec*rec)/(prec+rec));
                % ====================================
   
    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end

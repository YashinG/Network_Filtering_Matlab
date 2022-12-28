function [outCov, effectiveObs] = covMatWtd(inReturns, inTimeWts)
%CovMatWtd - Calculate weighted covariance matrix
%
%   Syntax:
%       outCov = CovMatWtd(inRets, inWts)
%    
%   Inputs:
%       inReturns     = (nDates x pAssets) matrix of stock returns
%       inTimeWts     = (nDates x 1) vector weight per observation

%
%   Outputs:
%       outCov          = (pAssets x pAssets) weighted covariance matrix
%       effectiveObs    = (scalar) Effective number of observations
%
%
%  Based on code from: Pozzi, F., Di Matteo, T. and Aste, T. (2012) 
%   ‘Exponential Smoothing Weighted Correlations’, The European Physical 
%   Journal B, 85(6), p. 175. 
%   Available at: https://doi.org/10.1140/epjb/e2012-20697-x.
%
%
%   Author: Yashin Gopi
%   Date: 11-Dec-2022; 

[nDates, pAssets] = size(inReturns); % n: number of observations; p: number of variables

% Check weight vector
% If no weight vector then use equal weight
if ~exist("inTimeWts","var")||isempty(inTimeWts)
    inTimeWts = ones(nDates,1)./nDates;
end

inTimeWts = inTimeWts/sum(inTimeWts); % Rescale wts to 100%

deameanRets   = inReturns - repmat(inTimeWts'*inReturns, nDates, 1); % Remove wtd mean
tempCov   = deameanRets'*(deameanRets.*repmat(inTimeWts, 1, pAssets)); % Calc wtd covariance 
outCov = 0.5 * (tempCov + tempCov'); % Ensure matrix is exactly symmetric

effectiveObs = 1/sum(inTimeWts.^2); % No need for numerator as wts sum to 100%

% Can convert to correlation using (a)
%   R = corrcov(C) 
% or (b)
%   R = diag(temp); % Variances
%   R = temp ./ sqrt(R * R'); % Matrix of Weighted Correlation Coefficients

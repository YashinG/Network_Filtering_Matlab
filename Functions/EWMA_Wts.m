function [outWts, effectiveObs] = EWMA_Wts(nDates, alpha)
%EWMA_Wts - Calculate EWMA wts
%
%   Syntax:
%       [outWts, effectiveObs] = EWMA_Wts(nDates, alpha)
%
%    
%   Inputs:
%       nDates = (scalar) Number of observations over which to determine wts
%       alpha  = (scalar) The exponential decay factor, and controls the 
%               amount of weight being applied to more recent observations.
%               Set == 0 for equal weights.
%
%   Outputs:
%       outWts          = (nDates x 1) Exponentialy weighted date points
%       effectiveObs    = (scalar) Effective sample size
%
%  See: Pozzi, F., Di Matteo, T. and Aste, T. (2012) 
%   ‘Exponential Smoothing Weighted Correlations’, The European Physical 
%   Journal B, 85(6), p. 175. 
%   Available at: https://doi.org/10.1140/epjb/e2012-20697-x.
%
%   w_t = w_0*e^{alpha(t-T)}
% 
% See: https://en.wikipedia.org/wiki/Design_effect#Effective_sample_size
%   For effective sample size:
%   n_eff = frac{(\sum_{i=1}^{n}w_{i})^{2}}{sum_{i=1}^{n}w_{i}^{2}}}}   
%
%   Author: Yashin Gopi
%   Date: 05-Jul-2022; 

if alpha == 0 % use equal time weights
    outWts = ones(nDates,1)./nDates;
else
    outWts = exp(alpha*((1:nDates)' - nDates)); % Calculate wts
end

outWts = outWts./sum(outWts); % Rescale to 100%

effectiveObs = 1/sum(outWts.^2); % No need for numerator as wts sum to 100%

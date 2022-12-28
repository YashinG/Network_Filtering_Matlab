function distMatrix = correlToDistMetric(inCorrMat)
% correlToDistMetric - converts the correlation matrix into a distance matrix
% in the spirit of Mantegna (1999), and Gower & Ross (1969) 
%   Syntax:
%       distMetric = correlToDistMetric(inCorr)    
%    
%   Inputs:
%       inCorrMat = (pAssets x pAssets) correlation matrix 
%       
%   Outputs:
%       distMatrix = (pAssets x pAssets) Distance matrix
%
%
%   Author: Yashin Gopi
%   Date: 11-Dec-2022;  

    distMatrix = sqrt(2*(1 - inCorrMat));
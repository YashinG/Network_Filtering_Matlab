function [finalRets, mktModeRets, checkMktMode, percExplMktMode] = preProcessRets(inReturns, inWts, inBlnStdize, inBlnRemMktMode)
%preProcessRets - pre-process returns (standardise) and remove market modes
%
%   Syntax:
%       strOut = preProcessRets(inReturns, inWts, in_nModes, in_blnStdize)
%
%    
%   Inputs:
%       inReturns           = (nDates x pAssets) matrix of stock returns
%       inWts               = (nDates x 1) vector weight per observation
%       inBlnStdize         = (== 1) to standardise/zscore the data (weighted)
%       inBlnRemMktMode     = (== 1) to remove PCA market mode (weighted)
%
%   Outputs:
%       finalRets          = (nDates x pAssets) matrix of processed stock returns
%       mktModeRets        = (nDates x 1) vector market mode returns
%       checkMktMode       = (scalar) Check if 1st PCA factor is market mode
%       percExplMktMode    = (scalar) Percentage var explained by market mode
%
%
%   Author: Yashin Gopi
%   Date: 11-Dec-2022; 

% Sample size and matrix dimension
[nDates, pAssets] = size(inReturns); 

% Check weight vector
% If no weight vector then use equal weight
if ~exist("inWts","var") || isempty(inWts)
    inWts = ones(nDates,1)./nDates;
end

if ~exist("inBlnStdize","var")||isempty(inBlnStdize)
    inBlnStdize = 0;
end

if ~exist("inBlnRemMktMode","var")||isempty(inBlnRemMktMode)
    inBlnRemMktMode = 0;
end

% Calculate weighted means and standard deviations
wtdMu = inWts'*inReturns;
wtdStd = std(inReturns,inWts);

% Standardize returns to use correlation as distance metric
% Use weighted metrics to account for different time weighting schemes (maybe EWMA)
if inBlnStdize == 1
    rets_1 = (inReturns - repmat(wtdMu, nDates, 1))./repmat(wtdStd, nDates, 1);
else
    rets_1 = inReturns; 
end

% Remove market mode
if inBlnRemMktMode == 1
    % Perform weighted PCA and take first factor as market mode
    [PCA_coeff, PCA_factorRets, ~, ~, PCA_Explained] = pca(rets_1,'Weights',inWts);
    
    checkMktMode = sum(PCA_coeff(:,1) > 0)/pAssets;
    percExplMktMode = PCA_Explained(1);
    
    % Check if the sign/direction of exposure to factor one is correct
    if checkMktMode < 0.5
        % More than half stocks have a negative exposure to factor one, then flip the sign 
        mktModeRets = -1.*PCA_factorRets(:,1);
        
    else
        mktModeRets = +1.*PCA_factorRets(:,1);
    end
 
    rets_2 = nan(nDates, pAssets);

    for iAsset = 1:pAssets
        % Perform weighted regression by multiplying column of ones and market mode returns by sqrt(wts)
        % See https://stackoverflow.com/questions/39315867/r-lm-result-differs-when-using-weights-argument-and-when-using-manually-rew
        temp_beta = regress(sqrt(inWts).*rets_1(:,iAsset), [sqrt(inWts).*ones(nDates,1), sqrt(inWts).*mktModeRets]);

        % Remove the market mode using the regression paramters
        rets_2(:,iAsset) = rets_1(:,iAsset) - (temp_beta(1) + temp_beta(2)*mktModeRets);
    end

else
    rets_2 = rets_1;
end

% Standardize returns again to use correlation as distance metric
% Calculate weighted means and standard deviations
wtdMu2 = inWts'*rets_2;
wtdStd2 = std(rets_2, inWts);

% Use weighted metrics to account for different time weighting schemes (maybe EWMA)
if inBlnStdize == 1
    finalRets = (rets_2 - repmat(wtdMu2, nDates, 1))./repmat(wtdStd2, nDates, 1);
else
    finalRets = rets_2; 
end

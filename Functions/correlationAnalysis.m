function stOut = correlationAnalysis(inReturns, inTimeWts, inNames, inBlnStdize, inBlnRemMktMode, inMethod, inDistXi, inPlotID)
% correlationAnalysis - Get the distribution of cross correlations in the correlation matrix
%
%   Syntax:
%       stOut = correlationAnalysis(inReturns, inTimeWts, inNames, inBlnStdize, inBlnRemMktMode, inMethod, inPlotID)
%
%   Inputs:
%       inReturns           = (nDates x pAssets) matrix of stock returns
%       inTimeWts           = (nDates x 1) vector of weight per observation (leave empty for equal wts)
%       inNames             = (1 x pAssets) cell array of tickers/names of stocks
%       inMethod            = (Scalar) Correlation calculaiton method (== 'QIS' or 'Normal')
%       inBlnStdize         = (== 1) to standardise/zscore the data (weighted)
%       inBlnRemMktMode     = (== 1) to remove PCA market mode (weighted)
%       inDistXi            = (1 x k) Pts at which to determine distribution of correlations (leave empty for automatic x's)
%       inplotID            = Plot figures
%
%   Outputs:
%       stOut                   = Structure containing all results
%       stOut.covMat            = (pAssets x pAssets) Resultant correlation matrix
%       stOut.corrMat           = (pAssets x pAssets) Resultant covariance matrix
%       stOut.corrValues        = ((pAssets x pAssets-1)/2 x 1) Vector of lower triangular entries in correlation matrix
%       stOut.corrDistSummary   = (1 x 5) Five number summary of correlations(quartiles)
%       stOut.corrAve           = (Scalar) Average correltion across the correlation matrix
%       stOut.corrFi            = (k x 1) Empirical distribution of correlations at each pt in inDistXi;
%       stOut.corrXi            = (k x 1) k equally spaced bins between -1 and +1;
%       stOut.inputs            = structure containing inputs for the function (for checking)
%       
%   Other m-files required:
%
%       QIS_Wtd.m - to use a weighted + QIS correlation matrix
%       covMatWtd.m - weighted covariance matrix
%       preProcessRets.m - standardise and remove market mode if required
%       Matlab Statistics and Machine Learning Toolbox  
%
%   Author: Yashin Gopi
%   Date: 11-Dec-2022; 

% sample size and matrix dimension
[nDates, pAssets] = size(inReturns);

% Check weight vector
% If no weight vector then use equal weight
if ~exist("inTimeWts","var")||isempty(inTimeWts)
    inTimeWts = ones(nDates,1)./nDates;
end

if ~exist("inDistXi","var")
    inDistXi = [];
end

% Pre-process returns (stdize or remove mtk mode)
processedRets = preProcessRets(inReturns, inTimeWts, inBlnStdize, inBlnRemMktMode);

% Perform correlation analysis
if strcmp(inMethod,'QIS')
    % Weighted correlation matrix with QIS shrinkage
    covMat = QIS_Wtd(processedRets, inTimeWts); % Ledoit Wolf Covariance

else
    % Standard weighted correlation matrix
    covMat = covMatWtd(processedRets, inTimeWts); % Covariance matrix

end

corrMat          = corrcov(covMat); % Matlab Stats Toolbox - convert to correl
corrValues       = corrMat(find(tril(ones(pAssets, pAssets),-1))); % Get on-diagonal correlation entries
[corrDistFi, corrDistXi] = ksdensity(corrValues, inDistXi); % Get emprirical density of correlations

% Store the 5 number summary of the correlations 
corrDistSummary = prctile(corrValues,[0, 25, 50, 75, 100]);     % Quartiles
corrAve         = mean(corrValues);     % Average correlation


% PLOTS -------------------------------------------
if inPlotID == 1
    figure
    plot(corrDistXi, corrDistFi);
    
    xline(corrAve,'--r', ['Average = ', num2str(round(corrAve,2))],'LabelOrientation','horizontal');

    if (inBlnStdize == 1)
        titleStdized = ', Stdized';
    else
        titleStdized = '';
    end

    if (inBlnRemMktMode == 1)
        titleMktMode = ', Market Mode Removed';
    else
        titleMktMode = '';
    end

    if all(inTimeWts) == 1/nDates
        titleTimeWts = '';
    else
        titleTimeWts = ' (Time Wtd)';
    end

    title(['Distribution of Correlations']);
    subtitle(['Method: ', inMethod, titleTimeWts, ...
        titleStdized,titleMktMode], 'Interpreter', 'none');

end

% Store results
stOut.covMat          = covMat;
stOut.corrMat         = corrMat;
stOut.corrValues      = corrValues;
stOut.corrFi          = corrDistFi;
stOut.corrXi          = corrDistXi;
stOut.corrDistSummary = corrDistSummary;
stOut.corrAve         = corrAve;

% if exist("fig", "var")
%     stOut.fig = fig;
% end

% Store all inputs for checking
stOut.inputs.inReturns       = inReturns;
stOut.inputs.processedRets   = processedRets;
stOut.inputs.corrMat         = corrMat;
stOut.inputs.inNames         = inNames;
stOut.inputs.inMethod        = inMethod;
stOut.inputs.inTimeWts       = inTimeWts;
stOut.inputs.inBlnStdize     = inBlnStdize;
stOut.inputs.inBlnRemMktMode = inBlnRemMktMode;
stOut.inputs.inPlotID        = inPlotID;



function stOut = correlationAnalysis_Dynamic(inReturns, inDates, inTimeWts, inRollWindow, inRollDateStep, inNames, inBlnStdize, inBlnRemMktMode, inMethod, inPlotID)
%correlationAnalysis_Dynamic - Get the distribution of cross correlations in the covariance matrix dynamically through time
%
%   Syntax:
%       stOut = correlationAnalysis_Dynamic(inReturns, inDates, inTimeWts, inRollWindow, inRollDateStep, inNames, inBlnStdize, inBlnRemMktMode, inMethod, inPlotID)
%
%   Inputs:
%       inReturns           = (nDates x pAssets) matrix of stock returns
%       inTimeWts           = (inRollWindow x 1) vector of weight per observation (leave empty for usual equal wts)
%       inNames             = (1 x pAssets) cell array of tickers/names of stocks
%       inMethod            = (Scalar) Correlation calculation method (== 'QIS' or 'Normal')
%       inBlnStdize         = (== 1) to standardise/zscore the data (weighted)
%       inBlnRemMktMode     = (== 1) to remove PCA market mode (weighted)
%       inRollWindow        = (Scalar) # observations to be used in correlation estimation for each window
%       inRollDateStep      = (Scalar) # observations to step forward through time
%       inplotID            = Plot figures
%
%   Outputs:
%       stOut                   = Structure containing all results through time (ober the nRollPts)
%       stOut.corrValues        = ((pAssets x pAssets-1)/2 x nRollPts) Matrix of lower triangular entries in correlation matrix
%       stOut.corrDistSummary   = (5 x nRollPts) Five number summary of correlations(quartiles)
%       stOut.corrAve           = (1 x nRollPts) Average correltion across the correlation matrix
%       stOut.dates             = (1 x nRollPts) Dates for each rolling point
%       stOut.processedRets     = {1 x nRollPts} cell array of the processed returns at each rolling point
%       stOut.inReturns         = {1 x nRollPts} cell array of the original returns at each rolling point
%       stOut.corrFi            = (41 x nRollPts) Empirical distribution of correlations evaluated at 41 equally spaced pts;
%       stOut.corrXi            = (41 x 1) 41 equally spaced pts between -1 and +1;
%       stOut.inputs            = structure containing inputs for the function (for checking)
%
%   Other m-files required:
%
%       QIS_Wtd.m - to use a weighted + QIS correlation matrix
%       covMatWtd.m - weighted covariance matrix
%       preProcessRets.m - standardise and remove market mode if required
%       correlationAnalysis.m - Get the distribution of cross correlations in the correlation matrix
%       Matlab Statistics and Machine Learning Toolbox  
%
%   Author: Yashin Gopi
%   Date: 11-Dec-2022; 

% sample size and matrix dimension
[nDates, pAssets] = size(inReturns);

% Check weight vector
% If no weight vector then use equal weight
if ~exist("inTimeWts","var")||isempty(inTimeWts)
    inTimeWts = ones(inRollWindow,1)./inRollWindow;
end

% Find dates at which to re-calc
%dateCount = 0;
maxDateCount = 0;

while (nDates - maxDateCount*(inRollDateStep)) >= inRollWindow
    maxDateCount = maxDateCount + 1;
    dateIndex(maxDateCount) = nDates - (maxDateCount - 1)*(inRollDateStep);
end

% Reverse dates into chronological order
dateIndex = flip(dateIndex);

stOutRollCorrel.corrDistFi      = nan(41, maxDateCount);
stOutRollCorrel.corrDistFi      = nan(41, maxDateCount);
stOutRollCorrel.corrAve         = nan(1, maxDateCount);
stOutRollCorrel.corrDistSummary = nan(5, maxDateCount);

figWaitbar = waitbar(0,'Running dynamic correlation analysis...');

%while (nDates - dateCount*(inRollDateStep)) >= inRollWindow
for iDate = 1:maxDateCount

    waitbar(iDate/maxDateCount, figWaitbar,'Running dynamic correlation analysis...');

    % Get date of calc and required returns
    tempDate     = inDates(dateIndex(iDate) ,1);
    tempRets     = inReturns(dateIndex(iDate) - inRollWindow +1: dateIndex(iDate) , :);

    % Get correlation matrix at rolling date point
    % stOut = correlationAnalysis(inReturns, inTimeWts, inNames, inBlnStdize, inBlnRemMktMode, inMethod, inDistXi, inPlotID)
    tempStOut = correlationAnalysis(tempRets, inTimeWts, inNames, inBlnStdize, inBlnRemMktMode, inMethod, [-1:0.05:1], 0);

    % Store data for roll period
    stOutRollCorrel.inReturns{iDate}       = tempStOut.inputs.inReturns;
    stOutRollCorrel.processedRets{iDate}   = tempStOut.inputs.processedRets;

    % Store results for roll period
    stOutRollCorrel.dates(iDate)        = tempDate; % Date
    stOutRollCorrel.corrValues(:,iDate) = tempStOut.corrValues;
    
    % Distribution of correlations
    stOutRollCorrel.corrDistFi(:,iDate)      = tempStOut.corrFi;
    stOutRollCorrel.corrAve(1,iDate)         = tempStOut.corrAve;
    stOutRollCorrel.corrDistSummary(:,iDate) = tempStOut.corrDistSummary;
    
end % iDate
close(figWaitbar)

stOutRollCorrel.corrDistXi = tempStOut.corrXi;

% Plot if required
if inPlotID == 1

    % Plot average correlation through time
    figure
    ax = axes();
    %plot(stOutRollCorrel.corrAve)
    [~, c] = contourf([1:maxDateCount],stOutRollCorrel.corrDistXi, stOutRollCorrel.corrDistFi);
    ax.CLim(1) = 0;
    ax.Colormap(1, :) = [1 1 1];

    colorbar
    
    % X-axis formatting
    % ID End of year dates
    [Y, M] = datevec(stOutRollCorrel.dates,'dd-mmm-yyyy');

    %idEOY = contains((stOutRollCorrel.dates),'Dec');
    idEOY = [Y(2:end) - Y(1:end-1); 0];
    xticks(find(idEOY));
    xticklabels(datestr(stOutRollCorrel.dates(find(idEOY)),'mmmyy'))
    xlim([1 maxDateCount]);
    xtickangle(90);

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

    if all(inTimeWts) == 1/inRollWindow
        titleTimeWts = '';
    else
        titleTimeWts = ' (Time Wtd)';
    end

    title(['Dynamic Correlations: Distribution']);
    subtitle(['Distance: ', inMethod, titleTimeWts, ...
        titleStdized,titleMktMode], 'Interpreter', 'none');

end

% Store results
stOut.dynamicCorrelations = stOutRollCorrel;

% Store all inputs for checking
stOut.inputs.inReturns       = inReturns;
stOut.inputs.inRollWindow    = inRollWindow;
stOut.inputs.inRollDateStep  = inRollDateStep;
stOut.inputs.inTimeWts       = inTimeWts;
stOut.inputs.inBlnStdize     = inBlnStdize;
stOut.inputs.inBlnRemMktMode = inBlnRemMktMode;
stOut.inputs.inDistMethod    = inMethod;
stOut.inputs.inPlotID        = inPlotID;



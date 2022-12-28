function stOut = getFilteredNetwork_Dynamic(inReturns, inDates, inTimeWts, inRollWindow, inRollDateStep, inNames, inNetworkFilter, inBlnStdize, inBlnRemMktMode, inDistMethod, inPlotID)
%getFilteredNetwork_Dynamic - Function to run the network filters through time
%
%   Syntax:
%       stOut = getFilteredNetwork_Dynamic(inReturns, inTimeWts, inRollWindow, inNames, inNetworkFilter, inBlnStdize, inBlnRemMktMode, inDistMethod, inSectors, inPlotID)
%
%
%   Inputs:
%       inReturns           = (n x p) matrix of stock returns
%       inWts               = (n x 1) vector weight per observation
%       inNames             = (1 x p) cell array of tickers/names of stocks
%       inSectors           = (1 x p) cell array of sector names of stocks
%       inDistMethod        = Distance method {'QIS_correlDistMetric','correlDistMetric','euclidean','cityblock',etc}
%       inplot_ID           = (== 1) to plot the network
%
%   Outputs:
%       stOut        =  Structure containing all results
%       stOut.S      = (p x p) Original similarity matrix
%       stOut.D      = (p x p) Original distance matrix
%       stOut.PMFG   = (p x p) PMFG matrix
%       stOut.PMFG_D = (p x p) PMFG filtered distance matrix;
%       stOut.PMFG_S = (p x p) PMFG filtered similarity matrix;
%
%
%
%   Other m-files required:
%
%     correlToDistMetric.m - to convert correlation matrix to a distance matrix
%     QIS_Wtd.m - to use a weighted + QIS correlation matrix
%

%
%   Author: Yashin Gopi
%   Date: 05-Jul-2022;


% sample size and matrix dimension
[nDates, pAssets] = size(inReturns);

% Check weight vector
% If no weight vector then use equal weight over rolling window length
if isempty(inTimeWts)||exist("inWts","var")
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

stOutRollNetwork.HybridCentrality = nan(pAssets, maxDateCount);
stOutRollNetwork.nTL              = nan(1, maxDateCount);
stOutRollNetwork.aveS             = nan(1, maxDateCount);
stOutRollNetwork.correlDist       = nan(20, maxDateCount);

figWaitbar = waitbar(0,'Running dynamic network filter...');

%while (nDates - dateCount*(inRollDateStep)) >= inRollWindow
for iDate = 1:maxDateCount

    waitbar(iDate/maxDateCount, figWaitbar,'Running dynamic network filter...');

    % Get date of calc and required returns
    tempDate     = inDates(dateIndex(iDate) ,1);
    tempRets     = inReturns(dateIndex(iDate) - inRollWindow +1: dateIndex(iDate) , :);

    % Get filtered network at rolling date point
    % stOut = getFilteredNetwork(inReturns, inTimeWts, inNames, inNetworkFilter, inBlnStdize, inBlnRemMktMode, inDistMethod, inPlotID)
    tempStOut = getFilteredNetwork(tempRets, inTimeWts, inNames, inNetworkFilter, inBlnStdize, inBlnRemMktMode, inDistMethod, 0);

    % Get graph metrics (length, centrality etc) using filtered distance matrix
    tempStOut.metrics = networkMetrics(graph(tempStOut.filteredNetwork_D, inNames), 'Distance');

    % Store data for roll period
    stOutRollNetwork.Data{iDate}        = tempStOut.inputs.inReturns;
    stOutRollNetwork.DataFinal{iDate}   = tempStOut.inputs.processedRets;

    % Store results for roll period
    stOutRollNetwork.dates(iDate)              = tempDate; % Date
    stOutRollNetwork.HybridCentrality(:,iDate) = tempStOut.metrics.node.Hybrid.XpY; % Hybrid Centrality
    stOutRollNetwork.nTL(1,iDate)              = tempStOut.metrics.tree.treeLenNorm; % Normalised Tree Length
    stOutRollNetwork.aveS(1,iDate)             = mean(tempStOut.S(logical(tril(ones(size(tempStOut.S)),-1)))); % Ave off diagonal entries in similarity matrix

    tempC = tempStOut.inputs.corrMat; % Correlation matrix
    % Distribution of correlations
    stOutRollNetwork.correlDist(:,iDate) = histcounts((tempC(find(tril(ones(size(tempC)),-1)))),[-1:0.1:1],'Normalization','probability');

end % iDate
close(figWaitbar)


% Plot if required
if inPlotID == 1

    % Plot NTL
    plot(stOutRollNetwork.nTL)

    % X-axis formatting
    xticks([1:maxDateCount]);
    xlim([1 maxDateCount]);
    idEOY = contains((stOutRollNetwork.dates),'Dec');

    xLabels = stOutRollNetwork.dates;
    xLabels(~idEOY) = {''};
    xticklabels(xLabels);
    xtickangle(90);


    if (inBlnStdize == 1)
        titleStdized = 'Stdized';
    else
        titleStdized = '';
    end

    if (inBlnRemMktMode == 1)
        titleMktMode = 'Market Mode Removed';
    else
        titleMktMode = '';
    end


    if all(inTimeWts) == 1/inRollWindow
        titleTimeWts = '';
    else
        titleTimeWts = '(Time Wtd)';
    end


    title(['Dynamic ' ,inNetworkFilter]);
    subtitle(['Distance: ', inDistMethod, titleTimeWts, ...
        ', ',titleStdized,...
        ', ',titleMktMode], 'Interpreter', 'none');

end

% Store results
stOut.dynamicNetwork = stOutRollNetwork;


% Store all inputs for checking
stOut.inputs.inReturns       = inReturns;
stOut.inputs.inNames         = inNames;
stOut.inputs.inTimeWts       = inTimeWts;
stOut.inputs.inRollWindow    = inRollWindow;
stOut.inputs.inRollDateStep  = inRollDateStep;
stOut.inputs.inBlnStdize     = inBlnStdize;
stOut.inputs.inBlnRemMktMode = inBlnRemMktMode;
stOut.inputs.inNetworkFilter = inNetworkFilter;
stOut.inputs.inDistMethod    = inDistMethod;
stOut.inputs.inPlotID        = inPlotID;

end

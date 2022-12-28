function stOut = getFilteredNetwork(inReturns, inTimeWts, inNames, inNetworkFilter, inBlnStdize, inBlnRemMktMode, inDistMethod, inPlotID)
% getFilteredNetwork - Wrapper function to run the MST function in MAtlab or the PMFG from Aste et al.
%
%   Syntax:
%       stOut = getFilteredNetwork(inReturns, inTimeWts, inNames, inNetworkFilter, inBlnStdize, inBlnRemMktMode, inDistMethod, inPlotID)
%
%
%   Inputs:
%       inReturns           = (nDates x pAssets) matrix of stock returns
%       inTimeWts           = (nDates x 1) vector of weight per observation (leave empty for equal wts)
%       inNames             = (1 x pAssets) cell array of tickers/names of stocks
%       inNetworkFilter     = (String) Filter to be used on the network (== 'PMFG' or 'MST')
%       inBlnStdize         = (== 1) to standardise/zscore the data (weighted)
%       inBlnRemMktMode     = (== 1) to remove PCA market mode (weighted)
%       inDistMethod        = (String) Distance method {'QIS_correlDistMetric','correlDistMetric','euclidean','cityblock',etc}
%       inPlotID            = (== 1) to plot the filtered network
%
%   Outputs:
%       stOut                   =  Structure containing all results
%       stOut.S                 = (pAssets x pAssets) Original similarity matrix
%       stOut.D                 = (pAssets x pAssets) Original distance matrix
%       stOut.filterName        = (String) Name of network filter used (MST or PMFG)
%       stOut.g                 = (graph object) Filtered network stored as a graph object
%       stOut.filteredNetwork   = (pAssets x pAssets) Sparse adjacency matrix from filtered network
%       stOut.filteredNetwork_D = (pAssets x pAssets) Filtered distance matrix
%       stOut.filteredNetwork_S = (pAssets x pAssets) Filtered similarity matrix
%       stOut.inputs            = structure containing inputs for the function (for checking)
%
%
%
%   Other m-files required:
%
%       correlToDistMetric.m - to convert correlation matrix to a distance matrix
%       QIS_Wtd.m - to use a weighted + QIS correlation matrix
%       covMatWtd.m - weighted covariance matrix
%       preProcessRets.m - standardise and remove market mode if required
%       Matlab Statistics and Machine Learning Toolbox
%
%       The 'pmfg' function from https://www.mathworks.com/matlabcentral/fileexchange/38689-pmfg
%       must be saved to the file path.
%
%       The 'getFilteredNetwork' function uses "matlab_bgl" package from
%       http://www.mathworks.com/matlabcentral/fileexchange/10922
%       and http://www.stanford.edu/~Edgleich/programs/matlab_bgl/ must be installed
%
%   Author: Yashin Gopi
%   Date: 05-Jul-2022;


% sample size and matrix dimension
[nDates, pAssets] = size(inReturns);

% Check weight vector
% If no weight vector then use equal weight
if ~exist("inTimeWts","var")||isempty(inTimeWts)
    inTimeWts = ones(nDates,1)./nDates;
end

% Set variable names as labels
labels = inNames';

% Pre-process returns (stdize or remove mtk mode)
processedRets = preProcessRets(inReturns, inTimeWts, inBlnStdize, inBlnRemMktMode);

% Determine distance matrix (D) and similarity matrix (S)
% Check distance method
if strcmp(inDistMethod,'correlDistMetric')
    % Weighted correlation matrix

    tempCov = covMatWtd(processedRets, inTimeWts); % Covariance matrix
    corrMat = corrcov(tempCov); % Matlab Stats Toolbox - convert to correl

    D = correlToDistMetric(corrMat); % Distance matrix
    S = 2-0.5*(D.^2); % Associated Similarity matrix

elseif strcmp(inDistMethod,'QIS_correlDistMetric')
    % Weighted correlation matrix with QIS shrinkage

    tempCov = QIS_Wtd(processedRets, inTimeWts); % Ledoit Wolf Covariance
    corrMat = corrcov(tempCov); % Matlab Stats Toolbox - convert to correl

    D = correlToDistMetric(corrMat); % Distance matrix
    S = 2-0.5*(D.^2); % Associated Similarity matrix
else

    tempCov = covMatWtd(processedRets, inTimeWts); % Covariance matrix
    corrMat = corrcov(tempCov); % Matlab Stats Toolbox - convert to correl

    % Use Matlab distance method
    D = squareform(pdist(processedRets',inDistMethod));

    % Convert to similarity using inverse ratio
    S = 1./(1 + squareform(D));
end

% Ensure matrices are exactly symmetric
S = 0.5 * (S + S');
D = 0.5 * (D + D');

if strcmp('MST', inNetworkFilter)

    % Determine MST and store as graph object
    stOut.g = minspantree(graph(D, labels));

elseif strcmp('PMFG', inNetworkFilter)

    % Determine PMFG and store as graph object
    sparse_S = sparse(S); % Convert S to correct format for PMFG function
    PMFG     = pmfg(sparse_S);  % Determine PMFG. Requires similarity matrix
    stOut.g  = graph(PMFG, labels); % Convert to matlab graph object

end

% Plot if required
if inPlotID == 1

    figure

    % plot the result in 2D
    h = plot(stOut.g, 'Layout','force');

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


    title(inNetworkFilter);
    subtitle(['Method: ', inDistMethod, titleTimeWts, ...
        titleStdized,titleMktMode], 'Interpreter', 'none');

end

% Outputs
stOut.S = S;
stOut.D = D;
stOut.filterName = inNetworkFilter;

stOut.filteredNetwork = adjacency(stOut.g,'weighted');

stOut.filteredNetwork_D = stOut.D;
stOut.filteredNetwork_D(full(stOut.filteredNetwork ==0)) = 0;

stOut.filteredNetwork_S = stOut.S;
stOut.filteredNetwork_S(full(stOut.filteredNetwork ==0)) = 0;


% Store all inputs for checking
stOut.inputs.inReturns       = inReturns;
stOut.inputs.processedRets   = processedRets;
stOut.inputs.corrMat         = corrMat;
stOut.inputs.inNames         = inNames;
stOut.inputs.inTimeWts       = inTimeWts;
stOut.inputs.inBlnStdize     = inBlnStdize;
stOut.inputs.inBlnRemMktMode = inBlnRemMktMode;
stOut.inputs.inNetworkFilter = inNetworkFilter;
stOut.inputs.inDistMethod    = inDistMethod;
stOut.inputs.inPlotID        = inPlotID;

end

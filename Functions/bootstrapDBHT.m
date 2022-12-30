function stOut = bootstrapDBHT(inReturns, inNames, inBlnStdize, inBlnRemMktMode, inLinkMethod, inDistMethod, inMaxClusters, in_nSim, inPlotID)
% bootstrapDBHT - Assess reliability of the number of clusters that results from the DBHT
%
%   Syntax:
%       stOut = bootstrapDBHT(inReturns, inNames, inBlnStdize, inBlnRemMktMode, inLinkMethod, inDistMethod, in_nSim, inPlotID)
%
%
%   Inputs:
%       inReturns           = (nDates x pAssets) matrix of stock returns
%       inNames             = (1 x pAssets) cell array of tickers/names of stocks
%       inLinkMethod        = Linkage method {'DBHT_PMFG', 'DBHT_TMFG','ward','single','complete','average'}
%       inBlnStdize         = (== 1) to standardise/zscore the data (weighted)
%       inBlnRemMktMode     = (== 1) to remove PCA market mode (weighted)
%       inDistMethod        = (String) Distance method {'QIS_correlDistMetric','correlDistMetric','euclidean','cityblock',etc}
%       inMaxClusters       = Maximum number of clusters to store data for in dendrogram
%       in_nSim             = (Scalar) Number of bootstrap samples to run
%       inPlotID            = (== 1) to plot the filtered network
%
%
%   Outputs:
%       stOut        =  Structure containing all results
%       stOut.S      = (pAssets x pAssets) Original similarity matrix
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
%
%   Author: Yashin Gopi
%   Date: 11-Dec-2022;



if ~strcmp(inLinkMethod, 'DBHT_PMFG') && ~strcmp(inLinkMethod, 'DBHT_TMFG')
    error('This function only works for the DBHT linkage functions.')
end

[nDates, pAssets] = size(inReturns);

% No EWMA weighting for bootstrapping
timeWts = ones(nDates,1)./nDates;

% Get full period DBHT results
% stOut = getClusters(inReturns, inTimeWts, inNames, inDistMethod, inLinkMethod,...
%       inBlnStdize, inBlnRemMktMode, inMaxClusters, inPlotID, ...
%       in_nClusters, inCompPartition)

tempStOut = getClusters(inReturns, timeWts, inNames, inDistMethod, ...
    inLinkMethod, inBlnStdize, inBlnRemMktMode, inMaxClusters,...
    0, [], []);

% Store full period DBHT
stOut.FullPeriod =  tempStOut;

% Get bootstrap samples (with replacement)
[~,bootsam]          = bootstrp(in_nSim,[], inReturns);
stOut.bootsam        = bootsam;
stOut.bootstrap.nSim = in_nSim;
stOut.nClusters      = nan(in_nSim, 1);

% Evaluate network for each sample

figWaitbar = waitbar(0,'Bootstraping Number of DBHT clusters...');

for iSample = 1:in_nSim
    waitbar(iSample/in_nSim, figWaitbar,'Bootstraping Number of DBHT clusters...');

    % Get temporary return data for sample
    tempRets = inReturns(bootsam(:,iSample),:);
    
    clear tempStOut

    % Generate clusters for bootstrap sample
    % stOut = getClusters(inReturns, inTimeWts, inNames, inDistMethod,
    %   inLinkMethod, inBlnStdize, inBlnRemMktMode, inMaxClusters,
    %   inPlotID, in_nClusters, inCompPartition)

    tempStOut = getClusters(tempRets, timeWts, inNames, inDistMethod, ...
        inLinkMethod, inBlnStdize, inBlnRemMktMode, [],...
        0, [], []);
    
    % Store number of clusters
    stOut.nClusters(iSample,1) = tempStOut.nClusters;

    clear tempStOut tempRets iSample

end
close(figWaitbar)

% Find average and std error of nClusters across bootstrap samples
stOut.nClusters_stdev       = std(stOut.nClusters);
stOut.nClusters_ave         = mean(stOut.nClusters);

if inPlotID == 1
    stOut.p = histogram(stOut.nClusters,'Normalization','probability');
    
    xlabel('nClusters');

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

    title('Bootstrap Estimate: Optimal Number of Clusters (DBHT)');
    subtitle(['Distance: ',inDistMethod, ...
        titleStdized,titleMktMode], 'Interpreter', 'none');

end

% Store all inputs for checking
stOut.inputs.inReturns       = inReturns;
stOut.inputs.inNames         = inNames;
stOut.inputs.inBlnStdize     = inBlnStdize;
stOut.inputs.inBlnRemMktMode = inBlnRemMktMode;
stOut.inputs.inLinkMethod    = inLinkMethod;
stOut.inputs.inDistMethod    = inDistMethod;
stOut.inputs.in_nSim         = in_nSim;
stOut.inputs.inPlot_ID       = inPlotID;


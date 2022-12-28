% Script to run hierachical clustering (DBHT, SLCA, ALCA, Ward, etc)

% Add paths to relevant folders

addpath('../Functions');
addpath('../External Functions');
addpath('../External Functions/PMFG')
addpath('../External Functions/matlab_bgl-4.0.1')

% Load data
load('../Data/Data.mat')

% Data inputs
stInputs.data.inReturns = stStockRets.rets;
stInputs.data.inNames   = stStockRets.aNames;

% Pre-processing inputs
stInputs.preProp.blnStdize     = 1;
stInputs.preProp.blnRemMktMode = 1;
stInputs.preProp.EWMA_Alpha   = 0; % {0, 0.005}

% Clustering inputs
stInputs.cluster.distanceMethod = 'QIS_correlDistMetric';  % {'QIS_correlDistMetric','correlDistMetric'}
stInputs.cluster.linkage        = 'ward';  % {'DBHT_PMFG','ward','single','complete','average','centroid','weighted','median'}
stInputs.cluster.maxClusters    = 15;
stInputs.cluster.nClusters      = 6;
stInputs.cluster.blnPlot        = 1;
stInputs.cluster.compPartition  = stStockRets.allSectorNum;

% Get EWMA weights
stInputs.data.timeWts = EWMA_Wts(size(stInputs.data.inReturns,1), stInputs.preProp.EWMA_Alpha);

% Combine share code and sector name to create new labels
stInputs.data.labels = strcat(stInputs.data.inNames, ' (',stStockRets.SuperSectorNames,')');


% Get clusters
% stOut = getClusters(inReturns, inTimeWts, inNames, inDistMethod, inLinkMethod,...
%   inBlnStdize, inBlnRemMktMode, inMaxClusters, inPlot_ID, in_nClusters)

stOutClusters = getClusters(stInputs.data.inReturns, stInputs.data.timeWts, ...
    stInputs.data.labels, stInputs.cluster.distanceMethod, stInputs.cluster.linkage,...
    stInputs.preProp.blnStdize, stInputs.preProp.blnRemMktMode, stInputs.cluster.maxClusters,...
    stInputs.cluster.blnPlot, stInputs.cluster.nClusters, stInputs.cluster.compPartition);


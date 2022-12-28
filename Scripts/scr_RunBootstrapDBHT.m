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


% Clustering inputs
stInputs.cluster.distanceMethod = 'QIS_correlDistMetric';  % {'QIS_correlDistMetric','correlDistMetric'}
stInputs.cluster.linkage        = 'DBHT_PMFG';  % {'DBHT_PMFG','DBHT_TMFG'}
stInputs.cluster.maxClusters    = 20;
stInputs.cluster.blnPlot        = 1;

% Bootstrap inputs
stInputs.bootstrap.nSim   = 10;

% Combine share code and sector name to create new labels
stInputs.data.labels = strcat(stInputs.data.inNames, ' (',stStockRets.SuperSectorNames,')');


% Get clusters
% stOut = bootstrapDBHT(inReturns, inNames, inBlnStdize, inBlnRemMktMode,...
%       inLinkMethod, inDistMethod, inMaxClusters, in_nSim, inPlotID)

stOutBootstrapDBHT = bootstrapDBHT(stInputs.data.inReturns, stInputs.data.labels, ...
    stInputs.preProp.blnStdize, stInputs.preProp.blnRemMktMode, ...
    stInputs.cluster.linkage, stInputs.cluster.distanceMethod, ...
    stInputs.cluster.maxClusters, stInputs.bootstrap.nSim, ...
    stInputs.cluster.blnPlot);


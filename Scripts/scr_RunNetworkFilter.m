% Script to run simple MST or PMFG and plot in Matlab

% Add paths to relevant folders

addpath('../Functions');
addpath('../External Functions/PMFG')
addpath('../External Functions/matlab_bgl-4.0.1/matlab_bgl')

% Load data
load('../Data/Data.mat')

% Data inputs
stInputs.data.inReturns = stStockRets.rets;
stInputs.data.inNames   = stStockRets.aNames;

% Pre-processing inputs
stInputs.preProp.blnStdize     = 1;
stInputs.preProp.blnRemMktMode = 1;
stInputs.preProp.EWMA_Alpha    = 0; % {0, 0.005}

% Network inputs
stInputs.network.distanceMethod = 'QIS_correlDistMetric';  % {'QIS_correlDistMetric','correlDistMetric'}
stInputs.network.filter          = 'PMFG';  % {'MST','PMFG'}
stInputs.network.blnPlot         = 1;


% Get EWMA weights
stInputs.data.timeWts = EWMA_Wts(size(stInputs.data.inReturns,1), stInputs.preProp.EWMA_Alpha);

% Run filter
% stOut = getFilteredNetwork(inReturns, inTimeWts, inNames, inNetworkFilter, ...
%   inBlnStdize, inBlnRemMktMode, inDistMethod, inPlotID)
    
stOutNetworkFilter = getFilteredNetwork(stInputs.data.inReturns, stInputs.data.timeWts, stInputs.data.inNames,...
         stInputs.network.filter, stInputs.preProp.blnStdize, stInputs.preProp.blnRemMktMode,...
         stInputs.network.distanceMethod, stInputs.network.blnPlot);

eval(['stOut_',stInputs.network.filter,'_', stInputs.network.distance_method,'= stOutNetworkFilter;']);
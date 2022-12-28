% Script to run dynamic MST or PMFG and plot in Matlab

% Add paths to relevant folders

addpath('../Functions');
addpath('../External Functions/PMFG')
addpath('../External Functions/matlab_bgl-4.0.1')

% Load data
load('../Data/Data.mat')

% Data inputs
stInputs.data.inReturns = stStockRets.rets;
stInputs.data.inNames   = stStockRets.aNames;
stInputs.data.inDates   = stStockRets.mlDates;

% Pre-processing inputs
stInputs.preProp.blnStdize     = 1;
stInputs.preProp.blnRemMktMode = 1;
stInputs.preProp.EWMA_Alpha   = 0; % {0, 0.005}

% Network inputs
stInputs.network.distance_method = 'QIS_correlDistMetric';  % {'QIS_correlDistMetric','correlDistMetric'}
stInputs.network.filter          = 'MST';  % {'MST','PMFG'}
stInputs.network.blnPlot         = 1;

% Rolling window inputs
stInputs.roll.window = 52*3; % 3 years
stInputs.roll.step = 4; % one month step

% Get EWMA weights
stInputs.data.timeWts = EWMA_Wts(stInputs.roll.window, stInputs.preProp.EWMA_Alpha);

% Run filter
%stOut = getFilteredNetwork_Dynamic(inReturns, inDates, inTimeWts, inRollWindow,...
% inRollDateStep, inNames, inNetworkFilter, inBlnStdize, inBlnRemMktMode, inDistMethod, inPlotID)

stOutNetworkFilter_Dynamic = getFilteredNetwork_Dynamic(...
    stInputs.data.inReturns, stInputs.data.inDates, stInputs.data.timeWts,...
    stInputs.roll.window, stInputs.roll.step, stInputs.data.inNames,...
    stInputs.network.filter, stInputs.preProp.blnStdize, stInputs.preProp.blnRemMktMode,...
    stInputs.network.distance_method, stInputs.network.blnPlot);


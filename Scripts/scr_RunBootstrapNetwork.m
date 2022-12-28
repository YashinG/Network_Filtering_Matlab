% Script to run bootstrap reliability esitmates for filtered networks

% Add paths to relevant folders

addpath('../Functions');
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

% Network inputs
% Network inputs
stInputs.network.distanceMethod  = 'QIS_correlDistMetric';  % {'QIS_correlDistMetric','correlDistMetric'}
stInputs.network.filter          = 'PMFG';  % {'MST','PMFG'}
stInputs.network.blnPlot         = 1;

% Bootstrap inputs
stInputs.bootstrap.nSim   = 10000;

% Run bootstrap
%  stOut = bootstrapNetwork(inReturns, inNames, inBlnStdize, inBlnRemMktMode, ...
%   inNetworkFilter, inDistMethod, in_nSim, inPlotID)

stOutBootstrapNetwork = bootstrapNetwork(stInputs.data.inReturns, stInputs.data.inNames, stInputs.preProp.blnStdize,...
        stInputs.preProp.blnRemMktMode, stInputs.network.filter, stInputs.network.distanceMethod,...
        stInputs.bootstrap.nSim, stInputs.network.blnPlot);

eval(['stOut_',stInputs.network.filter,'_', stInputs.network.distance_method,'.boostrap = stOutBootstrapNetwork;']);
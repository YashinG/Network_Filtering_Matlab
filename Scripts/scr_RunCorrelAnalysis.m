% Script to run correlation analysis
% Compare distribution of correlations over full period and dynamically
% Examine the impact of removing the market mode via PCA


% Add paths to relevant folders
addpath('../Functions');
addpath('../External Functions');

% Load data
load('../Data/Data.mat')

% Data inputs
stInputs.data.inReturns = stStockRets.rets;
stInputs.data.inNames   = stStockRets.aNames;
stInputs.data.inDates   = stStockRets.mlDates;

% Pre-processing inputs
stInputs.preProp.blnStdize     = 1;
%stInputs.preProp.blnRemMktMode = 1;
stInputs.preProp.EWMA_Alpha   = 0.005; % {0, 0.005}

% Clustering inputs
stInputs.correl.correlMethod = 'Normal';  % {'QIS','Normal'}

% Rolling window inputs
stInputs.roll.window = 52*3; % 3 years lookback period to estimate correlations
stInputs.roll.step   = 4; % one month step

% Get EWMA weights for each rolling window period
stInputs.roll.timeWts = EWMA_Wts(stInputs.roll.window, stInputs.preProp.EWMA_Alpha);

% ---------------- Static Analysis ----------------------------
% Get full period correlation distribution with and without removing market mode
% NB: Assume equal (standard) time weighting

% stOut = correlationAnalysis(inReturns, inTimeWts, inNames, inBlnStdize, inBlnRemMktMode, inMethod, inPlotID)

% Do not remove market mode
stOutCorrel.static.normal = correlationAnalysis(stInputs.data.inReturns, [], ...
    stInputs.data.inNames, stInputs.preProp.blnStdize, 0, 'Normal', [], 1);

% Do not remove market mode + QIS
stOutCorrel.static.normal_QIS = correlationAnalysis(stInputs.data.inReturns, [], ...
    stInputs.data.inNames, stInputs.preProp.blnStdize, 0, 'QIS', [], 1);


% Remove market mode
stOutCorrel.static.mktModeRem = correlationAnalysis(stInputs.data.inReturns, [], ...
    stInputs.data.inNames, stInputs.preProp.blnStdize, 1, 'Normal', [], 1);

% Remove market mode + QIS
stOutCorrel.static.mktModeRem_QIS = correlationAnalysis(stInputs.data.inReturns, [], ...
    stInputs.data.inNames, stInputs.preProp.blnStdize, 1, 'QIS', [], 1);


% ---------------- Dynamic Analysis ----------------------------
% Get correlation analysis dynamically 
% NB: Use EWMA time weights if required

% stOut = correlationAnalysis_Dynamic(inReturns, inDates, inTimeWts, inRollWindow,
%   inRollDateStep, inNames, inBlnStdize, inBlnRemMktMode, inMethod, inPlotID)

% Do not remove market mode
stOutCorrel.dynamic.normal = correlationAnalysis_Dynamic(stInputs.data.inReturns, stInputs.data.inDates, ...
    stInputs.roll.timeWts, stInputs.roll.window, stInputs.roll.step, stInputs.data.inNames, ...
    stInputs.preProp.blnStdize, 0, 'Normal', 1);

% Remove market mode
stOutCorrel.dynamic.mktModeRem = correlationAnalysis_Dynamic(stInputs.data.inReturns, stInputs.data.inDates, ...
    stInputs.roll.timeWts, stInputs.roll.window, stInputs.roll.step, stInputs.data.inNames, ...
    stInputs.preProp.blnStdize, 1, 'Normal', 1);



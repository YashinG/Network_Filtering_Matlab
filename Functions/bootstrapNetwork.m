function stOut = bootstrapNetwork(inReturns, inNames, inBlnStdize, inBlnRemMktMode, inNetworkFilter, inDistMethod, in_nSim, inPlotID)
% bootstrapNetwork - Assess reliability of network edges and network metrics
%
%   Syntax:
%       stOut = bootstrapNetwork(inReturns, inNames, inBlnStdize, inBlnRemMktMode, inNetworkFilter, inDistMethod, in_nSim, inPlotID)
%
%    
%   Inputs:
%       inReturns           = (nDates x pAssets) matrix of stock returns
%       inNames             = (1 x pAssets) cell array of tickers/names of stocks
%       inNetworkFilter     = (String) Filter to be used on the network (== 'PMFG' or 'MST')
%       inBlnStdize         = (== 1) to standardise/zscore the data (weighted)
%       inBlnRemMktMode     = (== 1) to remove PCA market mode (weighted)
%       inDistMethod        = (String) Distance method {'QIS_correlDistMetric','correlDistMetric','euclidean','cityblock',etc}
%       in_nSim             = (Scalar) Number of bootstrap samples to run
%       inPlotID            = (== 1) to plot the filtered network
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
%   Other m-files required: 
%       
%       correlToDistMetric.m - to convert correlation matrix to a distance matrix
%       QIS_Wtd.m - to use a weighted + QIS correlation matrix
%       covMatWtd.m - weighted covariance matrix
%       preProcessRets.m - standardise and remove market mode if required
%       Matlab Statistics and Machine Learning Toolbox  
%       getFilteredNetwork.m  - to run the MST or PMFG network filter  
%       networkMetrics.m      - to calculate the network topology metrics
%
%       The 'pmfg' function from https://www.mathworks.com/matlabcentral/fileexchange/38689-pmfg
%       must be saved to the file path.
%
%       The 'getFilteredNetwork' function uses "matlab_bgl" package from 
%       http://www.mathworks.com/matlabcentral/fileexchange/10922
%       and http://www.stanford.edu/~Edgleich/programs/matlab_bgl/ must be installed
%     
%   Author: Yashin Gopi
%   Date: 11-Dec-2022; 


[nDates ,pAssets] = size(inReturns);

% No EWMA weighting for bootstrapping
timeWts = ones(nDates,1)./nDates;

% Get full period network
% stOut = getFilteredNetwork(inReturns, inTimeWts, inNames, inNetworkFilter, ...
%   inBlnStdize, inBlnRemMktMode, inDistMethod, inPlotID)
tempStOut = getFilteredNetwork(inReturns, timeWts, inNames,...
         inNetworkFilter, inBlnStdize, inBlnRemMktMode,...
         inDistMethod, 0);

tempStOut.g = graph(tempStOut.filteredNetwork_D, inNames); % Get filtered distance matrix

% Store full period network
stOut.FullPeriod =  tempStOut;

% Get bootstrap samples (with replacement)
[~,bootsam]          = bootstrp(in_nSim,[], inReturns);
stOut.bootsam        = bootsam;
stOut.bootstrap.nSim = in_nSim;
stOut.checkOverlap   = nan(numedges(tempStOut.g), in_nSim);
stOut.XpY            = nan(pAssets, in_nSim);

% Evaluate network for each sample

figWaitbar = waitbar(0,'Running bootstrap resampling...');

for iSample = 1:in_nSim
     waitbar(iSample/in_nSim, figWaitbar,'Running bootstrap resampling...');
    
    % Get temporary return data for each sample
    tempRets = inReturns(bootsam(:,iSample),:);

    clear tempStOut
    
    % Generate filtered network for bootstrap sample

    % stOut = getFilteredNetwork(inReturns, inTimeWts, inNames, inNetworkFilter,...
    %   inBlnStdize, inBlnRemMktMode, inDistMethod, inSectors, inPlotID)

    tempStOut = getFilteredNetwork(tempRets, timeWts, inNames,...
         inNetworkFilter, inBlnStdize, inBlnRemMktMode,...
         inDistMethod, 0);
    
    tempStOut.g = graph(tempStOut.filteredNetwork_D, inNames);

    % Store edges in filtered network from sample
    %tempStOut.edgeData = table2array(tempStOut.g.Edges);

    % Calculate network metrics for sample
    tempStOut.metrics  = networkMetrics(tempStOut.g, 'Distance');

    % Store network metrics
    stOut.XpY(:, iSample) = tempStOut.metrics.node.Hybrid.XpY;

    % Check which of the original edges exist in the bootstrap sample
    stOut.checkOverlap(:,iSample) = findedge(tempStOut.g,...
        stOut.FullPeriod.g.Edges.EndNodes(:,1),...
        stOut.FullPeriod.g.Edges.EndNodes(:,2));

    clear tempStOut tempRets iSample

end
close(figWaitbar)

% Find percentage of occurences when edges still exist in bootstrap samples
stOut.edgeReliability = sum(stOut.checkOverlap > 0, 2)/in_nSim;

% Find average and std error of XpY bootstrap samples
stOut.XpY_stdev       = std(stOut.XpY,0,2);
stOut.XpY_ave         = mean(stOut.XpY,2);

if inPlotID == 1
    % Plot graph with realiblity factor overlaid onto edges
    figure
    p           = plot(stOut.FullPeriod.g,'Layout','force');
    p.LineWidth = 3*stOut.edgeReliability + eps;
    p.EdgeCData = stOut.edgeReliability;
    colormap jet
    colorbar
    
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


    title([inNetworkFilter, ' Bootstrap']);
    subtitle(['Distance: ',inDistMethod, ...
        titleStdized,titleMktMode], 'Interpreter', 'none');

end

% Store all inputs for checking
stOut.inputs.inReturns       = inReturns;
%stOut.inputs.processedRets   = processedRets;
stOut.inputs.inNames         = inNames;
stOut.inputs.inBlnStdize     = inBlnStdize;
stOut.inputs.inBlnRemMktMode = inBlnRemMktMode;
stOut.inputs.inNetworkFilter = inNetworkFilter;
stOut.inputs.inDistMethod    = inDistMethod;
stOut.inputs.in_nSim         = in_nSim;
stOut.inputs.inPlotID        = inPlotID;


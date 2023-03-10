function stOut = getClusters_Dynamic(inReturns, inDates, inTimeWts, inRollWindow, inRollDateStep, inNames, inDistMethod, inLinkMethod, inBlnStdize, inBlnRemMktMode, inMaxClusters, inPlotID, in_nClusters, inCompPartition)
%getClusters - Get hierarchical clustering for various linkages, including the DBHT method
%   Syntax:
%       stOut = getClusters(inReturns, inTimeWts, inNames, inDistMethod, inLinkMethod, inBlnStdize, inBlnRemMktMode, inMaxClusters, inPlot_ID, in_nClusters, in_comp)
%
%
%   Inputs:
%       inReturns           = (nDates x pAssets) matrix of stock returns
%       inWts               = (nDates x 1) vector weight per observation
%       inNames             = (1 x pAssets) cell array of tickers/names of stocks
%       inSectors           = (1 x pAssets) cell array of sector names of stocks
%       inDistMethod        = Distance method {'QIS_correlDistMetric','correlDistMetric','euclidean','cityblock',etc}
%       inLinkMethod        = Linkage method {'DBHT_PMFG', 'DBHT_TMFG','ward','single','complete','average'}
%       inMaxClusters       = Maximum number of clusters to store data for in dendrogram
%       inplot_ID           = Plot dendrogram
%       in_nClusters        = Specific number of clusters to store data for.
%                               Does not affect DBHT
%       inCompPartition     = (nPart x pAssets) Other partitions (such as an ICB/GICS classification for stocks) to compare to clusters generated here
%
%   Outputs:
%         stOut                         = Structure containing all results
%         stOut.allClusterIDs           = (pAssets x inMaxClusters-1) Clustering of stocks with original cluster numbers from 2 to max clusters;
%         stOut.leafOrder               = (pAssets x 1) Index containing the ordering of stocks according to sorted dendrogram;
%         stOut.clusterIDs              = (pAssets x 1) Clustering of stocks with original cluster numbers for specified or DBHT number of clusters;
%         stOut.nClusters               = (Scalar) Either user defined input or the DBHT outcome;
%         stOut.clusterIDsOrdered       = (pAssets x 1) Original order of stocks but using re-ordered cluster numbers according to dendrogram, for specified or DBHT number of clusters;
%         stOut.allClusterIDsOrdered    = (pAssets x inMaxClusters-1) Original order of stocks but using re-ordered cluster numbers, from 2 to max clusters;
%         stOut.tables
%               .allClusterIDsOrdered   = Data table that combines labels and cluster allocations for 2:max clusters
%               .clustersIDsOrdered     = Data table that combines labels and cluster allocations for user defined number of clusters, or DBHT number of clusters
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

% sample size and matrix dimension
[nDates, pAssets] = size(inReturns);

% Check weight vector
% If no weight vector then use equal weight over rolling window length
if ~exist("inTimeWts","var")||isempty(inTimeWts)
    inTimeWts = ones(inRollWindow,1)./inRollWindow;
end

% Set variable names as labels
labels = inNames';

% If user has not specified maximum number of clusters, then set == total assets
if ~exist("inMaxClusters","var") || isempty(inMaxClusters)
    inMaxClusters = pAssets;
end

% If number of clusters if not specified then set to inMaxClusters
% These inputs are NOT for the DBHT
if isempty(in_nClusters)
    in_nClusters = inMaxClusters;
end

if ~exist("inCompPartition","var")
    inCompPartition = [];
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

stOutRollClusters.Cluster      = nan(pAssets, maxDateCount);
stOutRollClusters.nClusters    = nan(1, maxDateCount);
if ~isempty(inCompPartition)
    nPartitions = size(inCompPartition,1);
    stOutRollClusters.adjRandIndex = nan(nPartitions, maxDateCount);
end

figWaitbar = waitbar(0,'Running dynamic network filter...');

% Adjust position for other progressbars
pos_figWaitbar   = get(figWaitbar,'position');
pos_figWaitbar_2 = [pos_figWaitbar(1) pos_figWaitbar(2)+ 2*pos_figWaitbar(4) pos_figWaitbar(3) pos_figWaitbar(4)];
set(figWaitbar,'position',pos_figWaitbar_2,'doublebuffer','on')

%while (nDates - dateCount*(inRollDateStep)) >= inRollWindow
for iDate = 1:maxDateCount

    waitbar(iDate/maxDateCount, figWaitbar,'Running dynamic clustering...');

    % Get date of calc and required returns
    tempDate     = inDates(dateIndex(iDate) ,1);
    tempRets     = inReturns(dateIndex(iDate) - inRollWindow +1: dateIndex(iDate) , :);

    % Get filtered network at rolling date point
    % Ignore ARI calc for now do it afterwards otherwise it will take too long
    % stOut = getClusters(inReturns, inTimeWts, inNames, inDistMethod, inLinkMethod, inBlnStdize, inBlnRemMktMode, inMaxClusters, inPlotID, in_nClusters, inCompPartition)
    tempStOut = getClusters(tempRets, inTimeWts, inNames, inDistMethod, inLinkMethod, inBlnStdize, inBlnRemMktMode, inMaxClusters, 0, in_nClusters, []);
           
    % Store data for roll period
    stOutRollClusters.inReturns{iDate}       = tempStOut.inputs.inReturns;
    stOutRollClusters.processedRets{iDate}   = tempStOut.inputs.processedRets;

    % Store results for roll period
    stOutRollClusters.dates(iDate)          = tempDate; % Date
    stOutRollClusters.clusterIDs(:,iDate)   = tempStOut.clusterIDs;
    stOutRollClusters.nClusters(1,iDate)    = max(tempStOut.clusterIDs);

    % ARI calc
    % Compare to user defined paritions (such as ICB/GICS) using Adj Rand Index
    if exist("inCompPartition","var") && ~isempty(inCompPartition)
        
        temp_ARI = nan(nPartitions, 1);

        for iPartition = 1:nPartitions
            % Calclate ARI
            temp_ARI(iPartition, 1) = RandIndex(tempStOut.clusterIDs', inCompPartition(iPartition,:));
        end
     
        stOutRollClusters.adjRandIndex(:,iDate) = temp_ARI;
    end
        
    tempC = tempStOut.inputs.corrMat; % Correlation matrix
    % Distribution of correlations
    stOutRollClusters.correlDist(:,iDate) = histcounts((tempC(find(tril(ones(size(tempC)),-1)))),[-1:0.1:1],'Normalization','probability');

end % iDate
close(figWaitbar)


% Plot if required
if inPlotID == 1
    figure

if exist("inCompPartition","var") && ~isempty(inCompPartition)
        tlt = tiledlayout(2, 1);

        nexttile
end
    % Plot n Clusters
    f1 = stairs(stOutRollClusters.nClusters,'-o');
    
     % X-axis formatting
    % ID End of year dates
    [Y, M] = datevec(stOutRollClusters.dates,'dd-mmm-yyyy');

    idEOY = [Y(2:end) - Y(1:end-1); 0];
    xticks(find(idEOY));
    xticklabels(datestr(stOutRollClusters.dates(find(idEOY)),'mmmyy'))
    xlim([1 maxDateCount]);
    xtickangle(90);

    tmp_yLim = get(gca,'YLim');
    set(gca,'YLim', [0 tmp_yLim(2)+1]);

    set(gca,'box','off')
    %grid on
    set(gca,'YGrid','on')

    ylabel('nClusters');

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

    if all(inTimeWts) == 1/inRollWindow
        titleTimeWts = '';
    else
        titleTimeWts = ' (Time Wtd)';
    end
    
    title(['Dynamic Number of Clusters: ',inLinkMethod], 'Interpreter', 'none');
    
    subtitle(['Distance: ', inDistMethod, titleTimeWts, ...
        titleStdized,titleMktMode], 'Interpreter', 'none');

    if exist("inCompPartition","var") && ~isempty(inCompPartition)
        nexttile
        f2 = plot(stOutRollClusters.adjRandIndex');
        
        xticks(find(idEOY));
        xticklabels(datestr(stOutRollClusters.dates(find(idEOY)),'mmmyy'))
        xlim([1 maxDateCount]);
        xtickangle(90);
        
        set(gca,'box','off')
        set(gca,'YGrid','on')
        title(['ARI: ',inLinkMethod], 'Interpreter', 'none');
        subtitle(['Distance: ', inDistMethod, titleTimeWts, ...
            titleStdized,titleMktMode], 'Interpreter', 'none');

        ylabel('ARI');
    end
end

% Store results
stOut.dynamicClusters = stOutRollClusters;

% Store all inputs for checking
stOut.inputs.inReturns       = inReturns;
%stOut.inputs.processedRets   = processedRets;
%stOut.inputs.corrMat         = corrMat;
stOut.inputs.labels          = labels;
stOut.inputs.inTimeWts       = inTimeWts;
stOut.inputs.inRollWindow    = inRollWindow;
stOut.inputs.inRollDateStep  = inRollDateStep;
stOut.inputs.inBlnStdize     = inBlnStdize;
stOut.inputs.inBlnRemMktMode = inBlnRemMktMode;
stOut.inputs.inDistMethod    = inDistMethod;
stOut.inputs.inLinkMethod    = inLinkMethod;
stOut.inputs.in_nClusters    = in_nClusters;
stOut.inputs.inMaxClusters   = inMaxClusters;
stOut.inputs.inCompPartition = inCompPartition;
stOut.inputs.inPlotID        = inPlotID;



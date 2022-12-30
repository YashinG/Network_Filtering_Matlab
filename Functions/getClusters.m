function stOut = getClusters(inReturns, inTimeWts, inNames, inDistMethod, inLinkMethod, inBlnStdize, inBlnRemMktMode, inMaxClusters, inPlotID, in_nClusters, inCompPartition)
%getClusters - Get hierarchical clustering for various linkages, including the DBHT method
%   Syntax:
%       stOut = getClusters(inReturns, inTimeWts, inNames, inDistMethod, inLinkMethod, inBlnStdize, inBlnRemMktMode, inMaxClusters, inPlot_ID, in_nClusters, in_comp)
%
%
%   Inputs:
%       inReturns           = (nDates x pAssets) matrix of stock returns
%       inTimeWts           = (nDates x 1) vector weight per observation
%       inNames             = (1 x pAssets) cell array of tickers/names of stocks
%       inSectors           = (1 x pAssets) cell array of sector names of stocks
%       inDistMethod        = Distance method {'QIS_correlDistMetric','correlDistMetric','euclidean','cityblock',etc}
%       inLinkMethod        = Linkage method {'DBHT_PMFG', 'DBHT_TMFG','ward','single','complete','average'}
%       inMaxClusters       = Maximum number of clusters to store data for in dendrogram
%       inplotID            = Plot dendrogram
%       in_nClusters        = Specific number of clusters to store data for.
%                               Does not affect DBHT
%       inCompPartition     = (nPart x pAssets) Other partitions (such as an ICB/GICS classification for stocks) to compare to clusters generated here
%
%   Outputs:
%         stOut                         = Structure containing all results
%         stOut.Z                       = (pAssets-1 x 3) Encodes a tree containing the hierarchical clusters
%         stOut.S                       = (pAssets x pAssets) Similarity matrix
%         stOut.D                       = (pAssets x pAssets) Distance matrix
%         stOut.leafOrder               = (pAssets x 1) Index containing the ordering of stocks according to the sorted dendrogram;
%         stOut.labels                  = (pAssets x 1) Cell array containing the labels for the tree
%         stOut.labelsOrdered           = (pAssets x 1) Cell array containing the sorted labels for the tree
%         stOut.clusterIDs              = (pAssets x 1) Clustering of stocks with original cluster numbers for specified or DBHT number of clusters;
%         stOut.clusterIDsOrdered       = (pAssets x 1) Original order of stocks but using re-ordered cluster numbers according to dendrogram, for specified or DBHT number of clusters; 
%         stOut.allClusterIDs           = (pAssets x inMaxClusters-1) Clustering of stocks with original cluster numbers from 2 to max clusters;
%         stOut.allClusterIDsOrdered    = (pAssets x inMaxClusters-1) Original order of stocks but using re-ordered cluster numbers, from 2 to max clusters;
%         stOut.nClusters               = (Scalar) Either user defined input or the DBHT outcome;
%         stOut.clusterIDsOrdered       = (nPart x inMaxClusters-1) Adj Rand Index (ARI) for partitions specified in 'inCompPartition' across variyng number of clusters
%         stOut.stOutDBHT               = Structure containing detailed output from DBHT function
%
%         stOut.tables
%               .clusterIDsOrdered      = Data table that combines labels and cluster allocations for user defined number of clusters, or DBHT number of clusters
%               .allClusterIDsOrdered   = Data table that combines labels and cluster allocations for 2:max clusters
%               .ARI                    = Data table shows the ARI for the specified paritions for 2:max clusters
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
% If no weight vector then use equal weight
if ~exist("inTimeWts","var")||isempty(inTimeWts)
    inTimeWts = ones(nDates,1)./nDates;
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

% Pre-process returns (stdize or remove mtk mode)
processedRets = preProcessRets(inReturns, inTimeWts, inBlnStdize, inBlnRemMktMode);

% Perform cluster analysis
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


% Check linkage method
if strcmp(inLinkMethod, 'DBHT_PMFG') || strcmp(inLinkMethod, 'DBHT_TMFG')
    % Determine DBHT

    % ---- different choices for D, S give different outputs!
    %D = squareform(D);
    %S = 2-D.^2/2;
    %S = 1./(1+D);
    if strcmp(inLinkMethod, 'DBHT_TMFG')
        DBHT_Method = 'TMFG';
    else
        DBHT_Method = 'PMFG';
    end

    [stOutDBHT.T8,stOutDBHT.Rpm,stOutDBHT.Adjv,...
        stOutDBHT.Dpm, stOutDBHT.Mv,stOutDBHT.Z] = DBHTs(D,S , DBHT_Method);% DBHT clustering

    stOutDBHT.Method = DBHT_Method;

    Z = stOutDBHT.Z;

    % Overwrite user inputted number of clusters with output from DBHT
    in_nClusters = length(unique(stOutDBHT.T8));

else
    % If not DBHT then do usual clustering methods
    Z = linkage(squareform(D), inLinkMethod);
end


% Reorder dendogram in more sensible manner.
leafOrder          = optimalleaforder(Z, squareform(D), 'Criteria', 'group');
[~, idx_leafOrder] = sort(leafOrder);

% Determine clusters and info across various "number of clusters"
% reqClusters          = 2:ceil((inMaxClusters + 1)/5)*5;
reqClusters          = 2:inMaxClusters;
nReqClusters         = size(reqClusters,2);
allClusterIDs        = nan(pAssets, nReqClusters);
allClusterIDsOrdered = nan(pAssets, nReqClusters);

% Find full hierarchy of stocks in clusters up to maximum number
for iReqCluster = 1:nReqClusters
    allClusterIDs(:,iReqCluster) = cluster(Z,'maxclust', reqClusters(1,iReqCluster));

    tempT = allClusterIDs(leafOrder,iReqCluster);
    newT = zeros(pAssets,1);
    newT(1,1) = 1;
    for iVar = 2:pAssets
        if tempT(iVar,1) == tempT(iVar-1,1)
            newT(iVar,1) = newT(iVar-1,1);
        else
            newT(iVar,1) = newT(iVar-1,1) + 1;
        end
    end
    allClusterIDsOrdered(:,iReqCluster) = newT(idx_leafOrder,1);

end

% Get main required number of clusters
T = cluster(Z,'maxclust', in_nClusters);

t = sort(Z(:,3));

%Find threshold for number of clusters in dendrogram
if in_nClusters > 1
    th = t(size(Z,1)+2-in_nClusters);
else
    th = t(size(Z,1))+0.001;
end
%th =0;

tempT = T(leafOrder);
newT = zeros(pAssets,1);
newT(1,1) = 1;
for iVar = 2:pAssets
    if tempT(iVar,1) == tempT(iVar-1,1)
        newT(iVar,1) = newT(iVar-1,1);
    else
        newT(iVar,1) = newT(iVar-1,1) + 1;
    end
end
orderedT = newT(idx_leafOrder,1);

% PLOTS -------------------------------------------
if inPlotID == 1

    figDendrogram = figure;
    % 1. Create dendrogram -------------------------------
    nexttile
    if inMaxClusters == 1
        [h1,~,outperm] = dendrogram(Z,0,'labels',labels,...
            'orient','top','colorthreshold',th, 'reorder',leafOrder);
        set(h1,'Color',[0 0.4470 0.7410])

    else
        [h1,~,outperm] = dendrogram(Z,0,'labels',strcat(labels,{' '},num2str(orderedT)),...
            'orient','right','colorthreshold',th, 'reorder',leafOrder);
    end

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

    title(['Hierarchical Clustering: ',inLinkMethod], 'Interpreter', 'none');
    subtitle(['Distance: ', inDistMethod, titleTimeWts, ...
        titleStdized,titleMktMode], 'Interpreter', 'none');
    
    %ylabel('Distance');
    xlabel('Distance');
    %xtickangle(90);
    set(findall(gcf,'-property','FontSize'),'FontSize',10)

end

% Compare to user defined paritions (such as ICB/GICS) using Adj Rand Index
if exist("inCompPartition","var") && ~isempty(inCompPartition)

    nPartitions = size(inCompPartition,1);
    adjRandIndex = nan(nPartitions, size(allClusterIDsOrdered,2));

    figWaitbar = waitbar(0,'Calculating ARI...');
    
    iCount = 0;
    nTotal = size(allClusterIDsOrdered,2)*nPartitions;

    for iPartition = 1:nPartitions
                
        for iCluster = 1:size(allClusterIDsOrdered,2)
            % Calclate ARI
            adjRandIndex(iPartition, iCluster) = RandIndex(allClusterIDsOrdered(:,iCluster)', inCompPartition(iPartition,:));
            
            % Update progress bar
            iCount = iCount + 1;
            waitbar(iCount/nTotal, figWaitbar,'Calculating ARI...');
        end
    end
    close(figWaitbar)

else
    adjRandIndex = [];
end

% Store all variable for output in a structure
stOut.Z = Z;
stOut.S = S;
stOut.D = D;

if exist('labels','var') && ~isempty(labels)

    stOut.labelsOrdered         = labels(leafOrder');
    stOut.labels                = labels;
    %[~,id_labels]               = ismember(labels, stOut.labelsOrdered);

end

if exist('stOutDBHT','var')
    stOut.stOutDBHT = stOutDBHT;
end

stOut.allClusterIDs        = allClusterIDs;
stOut.leafOrder            = leafOrder';
stOut.clusterIDs           = T;
stOut.nClusters            = in_nClusters;
stOut.clusterIDsOrdered    = orderedT;
stOut.allClusterIDsOrdered = allClusterIDsOrdered;
stOut.adjRandIndex         = adjRandIndex;

if exist('fig1','var')
    stOut.fig = figDendrogram;
end

% Store cluster allocations in tables
stOut.tables.allClusterIDsOrdered = array2table(stOut.allClusterIDsOrdered,'RowNames',stOut.labels,'VariableNames',strcat(string(2:inMaxClusters), ' Clusters'));
stOut.tables.clusterIDsOrdered    = array2table(stOut.clusterIDsOrdered,'RowNames',stOut.labels,'VariableNames',strcat(string(stOut.nClusters), ' Clusters'));

if exist("inCompPartition","var") && ~isempty(inCompPartition)
    stOut.tables.ARI = array2table(adjRandIndex,'RowNames',strcat('Partition_', string(1:nPartitions)),'VariableNames',strcat(string(2:inMaxClusters), ' Clusters'));
end

% Store all inputs for checking
stOut.inputs.inReturns       = inReturns;
stOut.inputs.processedRets   = processedRets;
stOut.inputs.corrMat         = corrMat;
stOut.inputs.labels          = labels;
stOut.inputs.inTimeWts       = inTimeWts;
stOut.inputs.inBlnStdize     = inBlnStdize;
stOut.inputs.inBlnRemMktMode = inBlnRemMktMode;
stOut.inputs.inDistMethod    = inDistMethod;
stOut.inputs.inLinkMethod    = inLinkMethod;
stOut.inputs.in_nClusters    = in_nClusters;
stOut.inputs.inMaxClusters   = inMaxClusters;
stOut.inputs.inCompPartition = inCompPartition;
stOut.inputs.inPlotID        = inPlotID;



function stOut = networkMetrics(in_G, weightType)
%networkMetrics - Calculates many relevant networks metrics (wtd and uwtd)
%
%   Syntax:
%       stOut = networkMetrics(in_G, weightType)
%
%   Description:
%       networkMetrics() - Calculates many relevant networks topology metrics (wtd and uwtd)
%
%   Inputs:
%       in_G        = Weighted matlab Graph object (must be weighted)
%       weightType  = Indicate whether weights are distances or correlations 
%                       (=='Distance' for distance matrix) or (=='Correlation' for correlation matrix)

%   Outputs:
%       stOut - Strucutre containing various network metrics (weighted and unweighted; for all nodes and the average)
%       stOut.info: Network info - edges and nodes
%       stOut.tree: Tree length info (treeLen, treeLenNorm)
%       stOut.node: Node metrics (.weighted, .unweighted)
%                   Metrics = degree, degreeCentrality, closeCentrality, betwCentrality, pagerankCentrality,
%                              EV_Centrality, shortestPaths, eccentricity
%       stOut.node: Node metrics (.Hyrbid)
%                   all_X, allRanks_X, all_Y, allRanks_Y, X, Y, XpY, XmY
%
%   Note - the variable "XpY" is the main metric of interest based on Pozzi, F., Di Matteo, T. and Aste, T. (2013)
%
%   Examples:
%       metrics = networkMetrics(MST.g, 'Distance');
%       metrics = networkMetrics(PMFG.g, 'Similarity');
%
%   Other m-files required: none
%
%   Based on: Pozzi, F., Di Matteo, T. and Aste, T. (2013)
%       'Spread of Risk Across Financial Markets: Better to Invest in the Peripheries',
%       Scientific Reports, 3(1), p. 1665. Available at: https://doi.org/10.1038/srep01665.

%   Author: Yashin Gopi
%   Date: 28-Jul-2022; Last revision: 28-Jul-2022
%
%   Copyright (c) 2022, Author
%   All rights reserved.

stOut.info.edges.Nodes  = in_G.Edges.EndNodes;
stOut.info.edges.Weight = in_G.Edges.Weight;
stOut.info.nodes        = in_G.Nodes;

if strcmp(weightType, 'Distance')
    % if weights are distances 
    stOut.info.edges.D = in_G.Edges.Weight; % Distances should be positive already
    stOut.info.edges.S = 2-0.5*(in_G.Edges.Weight).^2; % Convert distances to similarity using Mantegna (1999), and Gower & Ross (1969). 
else
    % if weights are correlations
    stOut.info.edges.S = 1 + in_G.Edges.Weight; % Make sure similarities are positive
    stOut.info.edges.D = sqrt(2*(1-in_G.Edges.Weight)); % Convert correlation to distances using Mantegna (1999), and Gower & Ross (1969). 
end

% Distance weighted graph
G_Dist = graph(in_G.Edges.EndNodes(:,1), in_G.Edges.EndNodes(:,2), stOut.info.edges.D, stOut.info.nodes);

% Similarity weighted graph
G_Sim = graph(in_G.Edges.EndNodes(:,1), in_G.Edges.EndNodes(:,2), stOut.info.edges.S, stOut.info.nodes);

% Calculate all shortest paths
tempD1    = distances(G_Dist,'Method','unweighted'); % use Distances
tempD1Wtd = distances(G_Dist,'Method','positive'); % use Distances


% Tree lengths - use distance matrix
stOut.tree.treeLen                         = sum(G_Dist.Edges.Weight);
stOut.tree.treeLenNorm                     = stOut.tree.treeLen/(numedges(G_Dist)-1);



% Get centrality metrics

% Unweighted metrics per node - note the use of S vs D
stOut.node.uw.degree                  = centrality(G_Sim, 'degree'); % Use Similarity
stOut.node.uw.degreeCentrality        = stOut.node.uw.degree/(numedges(G_Sim)-1); % Use Similarity

stOut.node.uw.closeCentrality         = centrality(G_Dist, 'closeness'); % Use Distance. No inversion like Pozzi. Higher == central
%stOut.node.uw.closeCentrality         = 1./(numnodes(G_Dist)*centrality(G_Dist, 'closeness')); % Use Distance. Also invert to align with Pozzi et al
stOut.node.uw.betwCentrality          = centrality(G_Dist, 'betweenness'); % Use Distance

stOut.node.uw.pagerankCentrality      = centrality(G_Sim, 'pagerank'); %  Use Similarity
stOut.node.uw.EV_Centrality           = centrality(G_Sim, 'eigenvector'); % Use Similarity

stOut.node.uw.shortestPaths           = tempD1;
stOut.node.uw.eccentricity            = 1./max(tempD1)'; % Higher == central

stOut.node.uw.all                     = [{"Degree","Degree Centrality","Close Centrality","Betw Centrality",...
                                        "Pagerank Centrality","EV Centrality","Eccentricity"};...
                                        num2cell([stOut.node.uw.degree, stOut.node.uw.degreeCentrality, ...
                                        stOut.node.uw.closeCentrality, stOut.node.uw.betwCentrality, ...
                                        stOut.node.uw.pagerankCentrality, stOut.node.uw.EV_Centrality, ...
                                        stOut.node.uw.eccentricity])];

% Average unweighted metrics
stOut.node.uw.ave.degree              = mean(stOut.node.uw.degree);
stOut.node.uw.ave.degreeCentrality    = mean(stOut.node.uw.degreeCentrality);
stOut.node.uw.ave.closeCentrality     = mean(stOut.node.uw.closeCentrality);
stOut.node.uw.ave.betwCentrality      = mean(stOut.node.uw.betwCentrality);
stOut.node.uw.ave.pagerankCentrality  = mean(stOut.node.uw.pagerankCentrality);
stOut.node.uw.ave.EV_Centrality       = mean(stOut.node.uw.EV_Centrality);
stOut.node.uw.ave.shortestPaths       = mean(tempD1(find(~tril(ones(size(tempD1))))));
stOut.node.uw.ave.eccentricity        = mean(stOut.node.uw.eccentricity);


% Weighted metrics per node -  - note the use of S vs D
stOut.node.wtd.degree                 = centrality(G_Sim, 'degree','Importance',G_Sim.Edges{:,'Weight'}); % Use Similarity
stOut.node.wtd.degreeCentrality       = stOut.node.wtd.degree/numedges(G_Sim); % Use Similarity

stOut.node.wtd.closeCentrality        = centrality(G_Dist, 'closeness','Cost',G_Dist.Edges{:,'Weight'}); % Use Distance. No inversion like Pozzi. Higher == central.
stOut.node.wtd.betwCentrality         = centrality(G_Dist, 'betweenness','Cost',G_Dist.Edges{:,'Weight'}); % Use Distance

stOut.node.wtd.pagerankCentrality     = centrality(G_Sim, 'pagerank','Importance',G_Sim.Edges{:,'Weight'}); % Use Similarity
stOut.node.wtd.EV_Centrality          = centrality(G_Sim, 'eigenvector','Importance', G_Sim.Edges{:,'Weight'}); % Use Similarity

stOut.node.wtd.shortestPaths          = tempD1Wtd;
stOut.node.wtd.eccentricity           = 1./max(tempD1Wtd)'; % Higher == central

stOut.node.wtd.all                     = [{"Degree","Degree Centrality","Close Centrality","Betw Centrality",...
                                        "Pagerank Centrality","EV Centrality","Eccentricity"};...
                                        num2cell([stOut.node.wtd.degree, stOut.node.wtd.degreeCentrality, ...
                                        stOut.node.wtd.closeCentrality, stOut.node.wtd.betwCentrality, ...
                                        stOut.node.wtd.pagerankCentrality, stOut.node.wtd.EV_Centrality, ...
                                        stOut.node.wtd.eccentricity])];

% Average weighted metrics
stOut.node.wtd.ave.degree             = mean(stOut.node.wtd.degree);
stOut.node.wtd.ave.degreeCentrality   = mean(stOut.node.wtd.degreeCentrality);
stOut.node.wtd.ave.closeCentrality    = mean(stOut.node.wtd.closeCentrality);
stOut.node.wtd.ave.betwCentrality     = mean(stOut.node.wtd.betwCentrality);
stOut.node.wtd.ave.pagerankCentrality = mean(stOut.node.wtd.pagerankCentrality);
stOut.node.wtd.ave.EV_Centrality      = mean(stOut.node.wtd.EV_Centrality);
stOut.node.wtd.ave.shortestPaths      = mean(tempD1Wtd(find(~tril(ones(size(tempD1Wtd))))));
stOut.node.wtd.ave.eccentricity       = mean(stOut.node.wtd.eccentricity);

% Hybrid centrality measure proposed by Pozzi, Di Matteo and Aste (2013)
%X, Y, XpY and XmY are, respectively, X, Y, (X + Y) and (X - Y)
% in the paper. A vertex characterized by high (low) ranking in terms
% of (X + Y) is likely to be a central (peripheral) vertex; a vertex
% characterized by high (low) ranking in terms of (X - Y) is likely
% to possess many unimportant (few important) connections. “High
% ranking” means “low score” (i.e. the most central vertex is assigned
% a small score). In detail:
%
% A small value of X indicates high connectedness whereas a large
% value indicates low connectedness
%
% A small value of Y indicates low eccentricity whereas a large
% value indicates high eccentricity
%
% A small value of XpY indicates high overall centrality whereas a
% large value indicates low overall centrality
%
% A small value of XmY indicates many low-quality connections
% whereas a large value indicates few high-quality connections


% Ranks for X metrics
% Flip high to low using minus sign
stOut.node.uw.ranks.degree          = tiedrank(-stOut.node.uw.degree);
stOut.node.wtd.ranks.degree         = tiedrank(-stOut.node.wtd.degree);
stOut.node.uw.ranks.betwCentrality  = tiedrank(-stOut.node.uw.betwCentrality);
stOut.node.wtd.ranks.betwCentrality = tiedrank(-stOut.node.wtd.betwCentrality);

stOut.node.Hybrid.all_X             = [{"Degree UW","Degree Centrality WTD", ...
                                "Betw Centrality UW", "Betw Centrality WTD"};...
                                 num2cell([stOut.node.uw.degree, stOut.node.wtd.degree,...
                                stOut.node.uw.betwCentrality, stOut.node.wtd.betwCentrality])];

stOut.node.Hybrid.allRanks_X        = [{"Degree UW","Degree Centrality WTD", ...
                                "Betw Centrality UW", "Betw Centrality WTD"};...
                                 num2cell([stOut.node.uw.ranks.degree, stOut.node.wtd.ranks.degree,...
                                stOut.node.uw.ranks.betwCentrality, stOut.node.wtd.ranks.betwCentrality])];

% Ranks for Y metrics
stOut.node.uw.ranks.eccentricity     = tiedrank(-stOut.node.uw.eccentricity);
stOut.node.wtd.ranks.eccentricity    = tiedrank(-stOut.node.wtd.eccentricity);
stOut.node.uw.ranks.closeCentrality  = tiedrank(-stOut.node.uw.closeCentrality);
stOut.node.wtd.ranks.closeCentrality = tiedrank(-stOut.node.wtd.closeCentrality);
stOut.node.uw.ranks.EV_Centrality    = tiedrank(-stOut.node.uw.EV_Centrality);
stOut.node.wtd.ranks.EV_Centrality   = tiedrank(-stOut.node.wtd.EV_Centrality);

stOut.node.Hybrid.allRanks_Y         = [{"Eccentricity UW","Eccentricity WTD", ...
                                "Close Centrality UW", "Close Centrality WTD",...
                                "EV UW", "EV WTD"};...
                                num2cell([stOut.node.uw.ranks.eccentricity, stOut.node.wtd.ranks.eccentricity,...
                                stOut.node.uw.ranks.closeCentrality, stOut.node.wtd.ranks.closeCentrality,...
                                stOut.node.uw.ranks.EV_Centrality, stOut.node.wtd.ranks.EV_Centrality])];

stOut.node.Hybrid.all_Y              = [{"Eccentricity UW","Eccentricity WTD", ...
                                "Close Centrality UW", "Close Centrality WTD",...
                                "EV UW", "EV WTD"};...
                                num2cell([stOut.node.uw.eccentricity, stOut.node.wtd.eccentricity,...
                                stOut.node.uw.closeCentrality, stOut.node.wtd.closeCentrality,...
                                stOut.node.uw.EV_Centrality, stOut.node.wtd.EV_Centrality])];

% Calcs for Y & X
stOut.node.Hybrid.X = (stOut.node.uw.ranks.degree + stOut.node.wtd.ranks.degree + ...
    stOut.node.uw.ranks.betwCentrality + stOut.node.wtd.ranks.betwCentrality - 4)./...
    (4*(numnodes(in_G)-1));

stOut.node.Hybrid.Y = (stOut.node.uw.ranks.eccentricity + stOut.node.wtd.ranks.eccentricity  + ...
    stOut.node.uw.ranks.closeCentrality + stOut.node.wtd.ranks.closeCentrality + ...
    stOut.node.uw.ranks.EV_Centrality + stOut.node.wtd.ranks.EV_Centrality - 6)./...
    (6*(numnodes(in_G)-1));

stOut.node.Hybrid.XpY = stOut.node.Hybrid.X + stOut.node.Hybrid.Y;
stOut.node.Hybrid.XmY = stOut.node.Hybrid.X - stOut.node.Hybrid.Y;

end

function sigmaHat = QIS_Wtd(inReturns, inTimeWts)
%QIS_Wtd - Calculates a wtd covariance matrix and then applies Ledoit-Wolf QIS shrinkage
%
%   Syntax:
%       outCov = QIS_Wtd(inReturns, inWts)
%
%   Inputs:
%       inReturns        = (nDates x pAssets) matrix of stock returns
%       inTimeWts        = (nDates x 1) vector weight per observation
%
%   Outputs:
%       sigmaHat          = (pAssets x pAssets) Wtd covariance matrix with QIS shrinkage
%
%   Other m-files required: CovMatWtd.m
%
%     Adapted from the the QIS.m shrinkage of Ledoit and Wolf:
%     https://www.mathworks.com/matlabcentral/fileexchange/106240-covshrinkage
%
%   Author: Yashin Gopi
%   Date: 11-Dec-2022;

[nDates, pAssets] = size(inReturns); % sample size and matrix dimension

% Check weight vector
% If no weight vector then use equal weight
if ~exist("inTimeWts","var")||isempty(inTimeWts)
    inTimeWts = ones(nDates,1)./nDates;
end

% Get weighted covariance matrix
[outCovWtd, effectiveObs] = covMatWtd(inReturns, inTimeWts);

k = 1; % Data was be demeaned in cov mat calc
N = effectiveObs; % use adjusted # of observations. For equal wts == nRows from inRets

%sample = (inRets' * inRets) ./ n; % sample covariance matrix
sample = outCovWtd; % use wtd covariance matrix

% ----- This code is from Ledoit-Wolf: QIS.m ---------------------

%%% EXTRACT sample eigenvalues sorted in ascending order and eigenvectors %%%
nDates = N - k; % adjust effective sample size
c = pAssets / nDates; % concentration ratio

[u, lambda] = eig(sample, 'vector'); % spectral decomposition
[lambda, isort] = sort(lambda); % sort eigenvalues in ascending order
u = u(:, isort); % eigenvectors follow their eigenvalues

%%% COMPUTE Quadratic-Inverse Shrinkage estimator of the covariance matrix %%%
h         = min(c^2, 1/c^2)^0.35 / pAssets^0.35; % smoothing parameter
invlambda = 1 ./ lambda(max(1, pAssets-nDates+1):pAssets); % inverse of (non-null) eigenvalues
Lj        = repmat(invlambda, [1, min(pAssets, nDates)])'; % like  1/lambda_j
Lj_i      = Lj - Lj'; % like (1/lambda_j)-(1/lambda_i)
theta     = mean(Lj.*Lj_i./(Lj_i.^2 + h^2 .* Lj.^2), 2); % smoothed Stein shrinker
Htheta    = mean(Lj.*(h .* Lj)./(Lj_i.^2 + h^2 .* Lj.^2), 2); % its conjugate
Atheta2   = theta.^2 + Htheta.^2; % its squared amplitude

if pAssets <= nDates % case where sample covariance matrix is not singular
    delta = 1 ./ ((1 - c)^2 * invlambda + 2 * c * (1 - c) * invlambda .* theta ...
        +c^2 * invlambda .* Atheta2);% optimally shrunk eigenvalues
else % case where sample covariance matrix is singular
    delta0 = 1 ./ ((c - 1) * mean(invlambda)); % shrinkage of null eigenvalues
    delta  = [repmat(delta0, [pAssets - nDates, 1]); 1 ./ (invlambda .* Atheta2)];
end

deltaQIS = delta .* (sum(lambda) / sum(delta)); % preserve trace
sigmaHat = u * diag(deltaQIS) * u'; % reconstruct covariance matrix
% ------------------------------------------------------------------------

sigmaHat = 0.5 * (sigmaHat + sigmaHat'); % Ensure matrix is exactly symmetric

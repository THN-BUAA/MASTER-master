function [PD, PF, AUC, MCC, Balance] = con_MSMDA_baseline(data, trainTarget, target,LOC)
% CON_MSMDA_BASELINE Summary of this function goes here
%   Detailed explanation goes here
% INPUTS:
%   (1) data - A cell consisting of all source datasets where each element is a (d+1)*n_i matrix.
%   (2) Traintarget - A (d+1)*n_t matrix, the last row is the label {0,1}
%   (3) target - A (d+1)*n_t matrix, the last row is the label {0,1}
% OUTPUTS:
%   
% 
% Reference: Z. Li, X. -Y. Jing, X. Zhu, H. Zhang, B. Xu and S. Ying, "On
%    the Multiple Sources and Privacy Preservation Issues for Heterogeneous
%    Defect Prediction," in IEEE Transactions on Software Engineering, vol.
%    45, no. 4, pp. 391-411, 1 April 2019, doi: 10.1109/TSE.2017.2780222.
%

Xl = trainTarget(1:end-1,:);
Yl = trainTarget(end,:);
Xu = target(1:end-1,:);
Yu = target(end,:);

% Normalization 
Xl = zscore(Xl,0,2); % zscore(X) is equal to zscore(X,0); 2 - zscore uses the means and standard deviations along the rows of X
Xu = zscore(Xu,0,2); 

Xl = Xl*diag(1./sqrt(sum(Xl.^2)));
Xu = Xu*diag(1./sqrt(sum(Xu.^2)));

% split training target data Xl into training data Xltr and validation data Xlva
nl = length(Yl);
nlt = ceil(nl/2);

rng('default')
ridx = randperm(nl);
Xltr = Xl(:,ridx(1:nlt));
Yltr = Yl(ridx(1:nlt));
Xlva = Xl(:,ridx(nlt+1:nl));
Ylva = Yl(ridx(nlt+1:nl));

Xl = [Xltr,Xlva];
Yl = [Yltr,Ylva];

% set predefined variables
options = [];
options.Xu = Xu;
options.Yu = Yu;
options.Xl = Xl;
options.Yl = Yl;

options.Xltr = Xltr; 
options.Yltr = Yltr;
options.Xlva = Xlva; 
options.Ylva = Ylva;

options.doTraining = 1; % training
options.Ws = [];


options.LOC = LOC;


% for differnt predcition combination
v = size(data,1);
mea = [];
for i=1:v
    source = data{i,1};
    % normalize source data
    [Xs,Ys] = normN2_source(source);
    Xs = Xs*diag(1./sqrt(sum(Xs.^2)));
    
    % evaluation
    mea = [mea; MDA(Xs,Ys,options)];
end

% sort the sources accroding to the performance measure 
g_measure = mea(:,2);
if sum(g_measure == 0)
    [smea,idx] = sortrows(mea,-3); % Sort rows in descending order acording to 3rd column elements (accuracy)
else
    [smea,idx] = sortrows(mea,[-2,-1]); % Sort rows in descending order acording to G_measure, if two or more rows have the same G-measure, sort them in descending order acording to AUC.
end
th = smea(1,2);

% realignment the multiple sources
sdata = data(idx,:);

% perform multiple source MDA
measure = MSMDA(sdata,options,th); % AUC, G-Measure, Accuracy

PD=measure(6); PF=measure(7); AUC=measure(1); MCC=measure(4); Balance=measure(5);

end

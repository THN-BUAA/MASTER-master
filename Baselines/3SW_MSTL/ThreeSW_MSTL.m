function [PD, PF, Precision, AUC, MCC, Balance, F1, probPos] = ThreeSW_MSTL(sources, target, LOC, LA_LD, k)
%THREESW_MSTL Summary of this function goes here
%   Detailed explanation goes here
% INPUTS:
%   (1) sources - A cell array where each element is a n_i*(n_feature+1) source dataset (the last column is the label, 0 - nondefective, 1 - defective)
%   (2) target  - A n_t*(n_feature+1) matrix where the last column is the label, 0 - nondefective, 1 - defective.
%   (3) LOC     - Line-of-code of each module. 
%   (4) LA_LD   - Sum of line-of-code-added and line-of-code-deleted
%   (5) k       - 
% OUTPUTS:
%
%
% Reference: Jiaojiao Bai, Jingdong Jia, Luiz Fernando Capretz, "A three-stage transfer 
%             learning framework for multi-source cross-project software defect prediction",
%            Information and Software Technology, Volume 150, 2022,106985,
%

%pyversion D:\Anaconda3\envs\py36\python.exe; % load python



%% Hyper-parameters
if ~exist('k', 'var')||isempty(k)
    k = 5; 
end
if ~exist('LOC', 'var')||isempty(LOC)
    LOC = []; 
end

nSrc = numel(sources);
if k>=nSrc
    k = 3;
    if k>=nSrc
        k = nSrc; %
    end
end

%% Top-k source datasets (Bellwether)
score = zeros(1, nSrc);
perf = zeros(nSrc*(nSrc-1),3);
count = 1;
for i=1:nSrc
    source = sources{i};
    targets = sources;
    %targets(i) = [];
    for j=1:numel(targets)
        if i==j
            continue;
        end
        test = targets{j};
        X = source(:,1:end-1);
        Y = source(:,end);
        LR = py.sklearn.linear_model.LogisticRegression(pyargs('class_weight','balanced', 'random_state', int64(0)));  % pyargs('random_state', int64(1), 'max_iter', int64(100))
        model = LR.fit(X, Y);
        testX = test(:,1:end-1);
        predLabel = model.predict(testX); % predLabel.data.double - a row vector
        cf=confusionmat(test(:,end),(predLabel.data.double)');
        TP=cf(2,2);
        TN=cf(1,1);
        FP=cf(1,2);
        FN=cf(2,1);
        PD=TP/(TP+FN);
        PF=FP/(FP+TN);
        G_measure = (2*PD*(1-PF))/(PD+1-PF);
        perf(count,:) = [i, j, G_measure];
        count = count + 1;
    end
end

for j=1:numel(targets)
    as_test = perf(perf(:,2)==j,:); % 
    [~, index] = sort(as_test(:,3), 'ascend');
    as_test = as_test(index, :);
    for i=1:size(as_test,1)
        score(as_test(i,1)) = score(as_test(i,1)) + i;
    end
end

[~, index] = sort(score, 'descend');
Bellwether = sources(index(1:k));

%% Kernel Mean Matching (KMM)
baseLearners = cell(1, k);
for i=1:numel(Bellwether)
    source = Bellwether{i};
    Xs = source(:,1:end-1);
    Xt = target(:,1:end-1);
    kmm = py.KMM.KMM(pyargs('kernel_type', 'rbf')); % Ensure that KMM.py is placed in the working directory
    beta = kmm.fit(Xs, Xt); % return a column vector (numpy.array type)
    Xs_new = repmat(beta.data.double, 1, size(Xs, 2)) .* Xs;
    Bellwether{i} = [Xs_new, source(:,end)]; 
    X = Xs_new;
    Y = source(:,end);
    LR = py.sklearn.linear_model.LogisticRegression(pyargs('class_weight','balanced', 'random_state', int64(0)));  % pyargs('random_state', int64(1), 'max_iter', int64(100))
    model = LR.fit(X, Y);
    baseLearners{i} = model;
end

%% wVote
clock_test_01 = tic;
nu = size(target,1);
Hs = zeros(nu, k);
probPosMatrix = zeros(nu, k);
for i=1:numel(baseLearners) % each base model
    model = baseLearners{i};
    Xt = target(:,1:end-1);
    predLabel = model.predict(Xt); % 
    Hs(:,i) = predLabel.data.double;
    probPos = model.predict_proba(Xt);
    probPos = probPos.data.double;
    probPosMatrix(:,i) = probPos(:,2);
end
Xt = target(:,1:end-1);
[temp ps] = mapminmax(Xt',0.001,1);
Xt = temp';
W = py.sklearn.metrics.pairwise.rbf_kernel(pyargs('X',py.numpy.asarray(Xt), 'gamma', int64(50))); % 
W = W.data.double;
D = diag(sum(W,2));
Lu = D-W;

% % Python
% P = Hs'*Lu*Hs;
% q = zeros(nu,1);
% G = -1*eye(nu);
% h = zeros(nu,1);
% A = ones(1, k);
% b = 1;
% py.cvxopt.solvers.options['show_progress'] = False;
%beta = py.cvxopt.solvers.qp(pyargs('P',py.cvxopt.matrix(P), 'q',py.cvxopt.matrix(q),'G',py.cvxopt.matrix(G),'h',py.cvxopt.matrix(h),'A',py.cvxopt.matrix(A),'b',py.cvxopt.matrix(b))); % Solve quadratic programming

% MATLAB: min 0.5*x'*H*x + f'*x   subject to:  A*x <= b; Aeq*x = beq; bl<=x<=br
H = roundn(2*Hs'*Lu*Hs, -4);
f = zeros(k,1);
Aeq = ones(1,k);
beq = 1;
lb = zeros(k,1);
options = optimoptions('quadprog','Display','off');
beta = quadprog(H,f,[],[],Aeq,beq,lb,[],[],options); % return a column vector, ,options = optimoptions('Display','off')

%% Prediction performance
probPos = probPosMatrix*beta;
try
    [PD,PF,AUC,MCC,Balance,F1] = Performance(target(:,end),probPos); %
catch
    PD=nan; PF=nan; F1=nan; AUC=nan; MCC=nan; Balance=nan;
end

% if ~isempty(LOC)
    % Popt20 = CalculatePopt(target(:,end), double(probPos>0.5), LOC);
    % Popt = CalculatePopt(target(:,end), double(probPos>0.5), LOC, 1);
% else
    % Popt20 = nan;
    % Popt = nan;
% end



end


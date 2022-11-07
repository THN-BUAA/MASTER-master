% % % 
%  an example of MSMDA to predict CM1 project by using other heterogeneous source projects
% % % 

clear all
close all
clc

addpath('.\utility\');
addpath('.\liblinear\');

% load target data
load('CM1.mat')
target_data = target{1,1};
target_name = target{1,2};
target_ridx = target{1,3}; % ÿһ�е�ǰ42����1:42��42(���ݼ���42��������)�������������

% load source data
load('CM1_source.mat');
    
Rep = 10; % runs
ratio = 0.1; % labled ratio

result = [];
for loop = 1:Rep
    measure = con_MSMDA(data,target_ridx(loop,:),target_data,ratio);
    % result = [result; measure(1:2)];
    result = [result; measure];
end

save RESULT.mat result
disp('done !')

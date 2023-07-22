function [PD, PF, AUC, MCC, Balance, F1] = DBN_CP_MS(test, csvSrcPath, targetPath, str_i, srcCodeDir, tarCodeDir, saveDir, LOC, LA_LD)
%DBN_CP_MS Summary of this function goes here
%   Detailed explanation goes here


trainTime = 0;
testTime = 0;
probPoss = [];
for k=1:numel(csvSrcPath)
    [ PD_dbn,PF_dbn, AUC_dbn, MCC_dbn, Balance_dbn, F1_dbn, probPos] = DBN_CP(csvSrcPath{k}, targetPath, str_i, srcCodeDir, tarCodeDir, saveDir);
    probPoss = [probPoss, probPos];
end
probPos_dbn = mean(probPoss, 2);

try
    [PD,PF,AUC,MCC,Balance,F1] = Performance(test(:,end), probPos_dbn); % Call self-defined Performance()
catch
    PD=nan;PF=nan;F1=nan;AUC=nan;MCC=nan;Balance=nan;
end

Popt20 = CalculatePopt(test(:,end), probPos_dbn, LOC);
Popt = CalculatePopt(test(:,end), probPos_dbn, LOC, 1);

% if ~exist('LA_LD','var')||isempty(LA_LD)
    % IFA = IFAfun(test(:,end), probPos_dbn);
% else
    % IFA = IFAfun(test(:,end), probPos_dbn, LA_LD);
% end
        
end


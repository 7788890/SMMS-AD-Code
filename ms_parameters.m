clc;clear;
stats = load('60sec\5ms\HC.mat').outputStats;
n=length(stats);
%group = {stats.Group};
MeanDuration = reshape([stats.MeanDuration],5,n);
MeanDuration = normalize(MeanDuration);
MeanOccurrence = reshape([stats.MeanOccurrence],5,n);
MeanOccurrence = normalize(MeanDuration);
Coverage = reshape([stats.Coverage],5,n);
Coverage = normalize(Coverage);
OrgTM = reshape([stats.OrgTM],5,5,n);
OrgTM = reshape(OrgTM,25,n);
OrgTM = [OrgTM(2:6,:);OrgTM(8:12,:);OrgTM(14:18,:);OrgTM(20:24,:)];

msfeature = [MeanDuration;MeanOccurrence;Coverage;OrgTM];
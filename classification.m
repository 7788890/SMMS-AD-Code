    clc;clear;
rng(42);

subject_ids_AD = load("60sec\label\AD_label.mat").sub_idx;
subject_ids_FT = load("60sec\label\FTD_label.mat").sub_idx;
subject_ids_HC = load("60sec\label\HC_label.mat").sub_idx;

subject_ids_ADFT = [subject_ids_AD,subject_ids_FT];


%% 
ADfeatures_ms = load('60sec\ms\AD_ms.mat').msfeature;
FTfeatures_ms = load('60sec\ms\FTD_ms.mat').msfeature;
HCfeatures_ms = load('60sec\ms\HC_ms.mat').msfeature;

%% 
% AD_01_04_wpli = load("D:\dataset\AD\subband\AD_01_04_wpli.mat").wpli_AD;
% AD_04_08_wpli = load("D:\dataset\AD\subband\AD_04_08_wpli.mat").wpli_AD;
% AD_08_13_wpli = load("D:\dataset\AD\subband\AD_08_13_wpli.mat").wpli_AD;
% AD_13_30_wpli = load("D:\dataset\AD\subband\AD_13_30_wpli.mat").wpli_AD;
% AD_wpli = [AD_01_04_wpli;AD_04_08_wpli;AD_08_13_wpli;AD_13_30_wpli];
% 
% FT_01_04_wpli = load("D:\dataset\AD\subband\FT_01_04_wpli.mat").wpli_FT;
% FT_04_08_wpli = load("D:\dataset\AD\subband\FT_04_08_wpli.mat").wpli_FT;
% FT_08_13_wpli = load("D:\dataset\AD\subband\FT_08_13_wpli.mat").wpli_FT;
% FT_13_30_wpli = load("D:\dataset\AD\subband\FT_13_30_wpli.mat").wpli_FT;
% FT_wpli = [FT_01_04_wpli;FT_04_08_wpli;FT_08_13_wpli;FT_13_30_wpli];
% 
% HC_01_04_wpli = load("D:\dataset\AD\subband\HC_01_04_wpli.mat").wpli_HC;
% HC_04_08_wpli = load("D:\dataset\AD\subband\HC_04_08_wpli.mat").wpli_HC;
% HC_08_13_wpli = load("D:\dataset\AD\subband\HC_08_13_wpli.mat").wpli_HC;
% HC_13_30_wpli = load("D:\dataset\AD\subband\HC_13_30_wpli.mat").wpli_HC;
% HC_wpli = [HC_01_04_wpli;HC_04_08_wpli;HC_08_13_wpli;HC_13_30_wpli];



%%
ADbrain_region_ave = load('60sec\brain_region_wpli\ADave.mat').avg_values;
FTbrain_region_ave = load('60sec\brain_region_wpli\FTave.mat').avg_values;
HCbrain_region_ave = load('60sec\brain_region_wpli\HCave.mat').avg_values;


%%
AD_r2r = load('60sec\brain_region_wpli\ADwpli_r2r.mat').result_matrix';
FT_r2r = load('60sec\brain_region_wpli\FTwpli_r2r.mat').result_matrix';
HC_r2r = load('60sec\brain_region_wpli\HCwpli_r2r.mat').result_matrix';

%%
ADfeatures = [normalize(ADfeatures_ms);normalize(ADbrain_region_ave);normalize(AD_r2r)];
FTfeatures = [normalize(FTfeatures_ms);normalize(FTbrain_region_ave);normalize(FT_r2r)];
HCfeatures = [normalize(HCfeatures_ms);normalize(HCbrain_region_ave);normalize(HC_r2r)];

ADFTfeatures = [ADfeatures,FTfeatures];


%% 创建标签向量 (1=AD, 2=FTD, 3=HC)
labels_AD = ones(1, size(subject_ids_AD, 2));
labels_FT = 2 * ones(1, size(subject_ids_FT, 2));
labels_HC = 3 * ones(1, size(subject_ids_HC, 2));

labels_ADFT = 4 * ones(1, size(subject_ids_AD, 2) + size(subject_ids_FT, 2));

all_features = [ADFTfeatures, HCfeatures ]';
all_subject_ids = [subject_ids_ADFT, 99 + subject_ids_HC];
all_labels = [labels_ADFT, labels_HC];

%%
% 使用ReliefF进行特征排序
[ranks, weights] = relieff(all_features, all_labels, 10, 'method', 'classification');

% 转置以便于处理 (现在每行是一个样本)
all_labels = all_labels';
all_subject_ids = all_subject_ids';

fprintf('总样本数: %d\n', length(all_labels));
fprintf('唯一受试者数: %d\n', length(unique(all_subject_ids)));
fprintf('各类别样本数 - FTD: %d, HC: %d\n', ...
    sum(all_labels==2), sum(all_labels==3));

% 设置不同的特征选择数量
feature_nums = 1:size(all_features,2);

% 存储每个特征数量的结果
mean_accuracies = zeros(length(feature_nums), 1);
std_accuracies = zeros(length(feature_nums), 1);
all_results = cell(length(feature_nums), 1);

% 外层10折交叉验证参数
outer_k_folds = 10;

fprintf('\n=== 外层10折交叉验证 + 70/30划分 ===\n');

% 对每个特征数量进行外层交叉验证
for i = 1:length(feature_nums)
    num_features = feature_nums(i);
    fprintf('\n测试特征数量: %d\n', num_features);
    
    % 选择前N个最重要的特征
    selected_features = all_features(:, ranks(1:min(num_features, size(all_features, 2))));
    
    % 存储外层交叉验证的结果
    outer_accuracies = zeros(outer_k_folds, 1);
    
    % 外层10折交叉验证
    outer_cv = cvpartition(all_labels, 'KFold', outer_k_folds);
    
    for outer_fold = 1:outer_k_folds
        fprintf('  外层折 %d/%d...', outer_fold, outer_k_folds);
        
        % 获取当前外层折的训练集和测试集
        train_idx = outer_cv.training(outer_fold);
        test_idx = outer_cv.test(outer_fold);
        
        % 当前外层折的数据
        outer_train_features = selected_features(train_idx, :);
        outer_train_labels = all_labels(train_idx);
        outer_train_subjects = all_subject_ids(train_idx);
        
        % 在外层训练集上进行70/30划分
        try
            results = stratified_subject_level_split(outer_train_features, outer_train_labels, outer_train_subjects, 'svm');
            outer_accuracies(outer_fold) = results.accuracy * 100;
            fprintf('准确率: %.2f%%\n', outer_accuracies(outer_fold));
        catch ME
            fprintf('错误: %s\n', ME.message);
            outer_accuracies(outer_fold) = 0;
        end
    end
    
    % 计算均值和标准差
    mean_accuracies(i) = mean(outer_accuracies);
    std_accuracies(i) = std(outer_accuracies);
    all_results{i} = outer_accuracies;
    
    fprintf('特征数量 %d - 平均准确率: %.2f%% ± %.2f%%\n', ...
        num_features, mean_accuracies(i), std_accuracies(i));
end

% 绘制特征选择数量与准确率的关系图
figure('Position', [100, 100, 1000, 600]);

% 准确率曲线
errorbar(feature_nums, mean_accuracies, std_accuracies, 'o-', 'LineWidth', 2, ...
    'MarkerSize', 8, 'MarkerFaceColor', 'blue', 'Color', 'blue', ...
    'CapSize', 10);
xlabel('特征选择数量', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('准确率 (%)', 'FontSize', 12, 'FontWeight', 'bold');
title('特征选择数量 vs 分类准确率 (外层10折CV + 70/30划分)', 'FontSize', 14, 'FontWeight', 'bold');
grid on;

% 找到最佳准确率
[best_accuracy, best_idx] = max(mean_accuracies);
best_num_features = feature_nums(best_idx);
best_std = std_accuracies(best_idx);

% 标记最佳点
hold on;
plot(best_num_features, best_accuracy, 'ro', 'MarkerSize', 12, 'LineWidth', 3);
text(best_num_features, best_accuracy + std_accuracies(best_idx) + 1, ...
    sprintf('最佳: %d特征\n%.2f%%±%.2f%%', best_num_features, best_accuracy, best_std), ...
    'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold', ...
    'BackgroundColor', 'white', 'EdgeColor', 'black');

% 设置坐标轴范围
xlim([min(feature_nums)-1, max(feature_nums)+1]);
ylim([max(0, min(mean_accuracies)-5), min(max(mean_accuracies)+5, 100)]);

% 添加数据标签
for i = 1:length(feature_nums)
    if mod(i,2) == 0 % 每隔一个点显示标签，避免重叠
        text(feature_nums(i), mean_accuracies(i) - std_accuracies(i) - 1.5, ...
            sprintf('%.1f%%', mean_accuracies(i)), ...
            'HorizontalAlignment', 'center', 'FontSize', 8);
    end
end

% 显示最佳结果摘要
fprintf('\n=== 最佳结果摘要 ===\n');
fprintf('最佳特征数量: %d\n', best_num_features);
fprintf('最佳准确率: %.2f%% ± %.2f%%\n', best_accuracy, best_std);

% 显示详细的外层交叉验证结果
fprintf('\n=== 外层10折交叉验证详细结果 ===\n');
for i = 1:length(feature_nums)
    fprintf('特征数量 %d: ', feature_nums(i));
    fprintf('%.1f±%.1f%% [', mean_accuracies(i), std_accuracies(i));
    for j = 1:length(all_results{i})
        fprintf('%.1f', all_results{i}(j));
        if j < length(all_results{i}), fprintf(', '); end
    end
    fprintf(']\n');
end
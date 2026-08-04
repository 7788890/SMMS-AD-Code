clc; clear;
rng(42);

subject_ids_AD = load("60sec\label\AD_label.mat").sub_idx;
subject_ids_FT = load("60sec\label\FTD_label.mat").sub_idx;
subject_ids_HC = load("60sec\label\HC_label.mat").sub_idx;

ADfeatures_ms = load('60sec\ms\AD_ms.mat').msfeature;
FTfeatures_ms = load('60sec\ms\FTD_ms.mat').msfeature;
HCfeatures_ms = load('60sec\ms\HC_ms.mat').msfeature;

ADbrain_region_ave = load('60sec\brain_region_wpli\ADave.mat').avg_values;
FTbrain_region_ave = load('60sec\brain_region_wpli\FTave.mat').avg_values;
HCbrain_region_ave = load('60sec\brain_region_wpli\HCave.mat').avg_values;

AD_r2r = load('60sec\brain_region_wpli\ADwpli_r2r.mat').result_matrix';
FT_r2r = load('60sec\brain_region_wpli\FTwpli_r2r.mat').result_matrix';
HC_r2r = load('60sec\brain_region_wpli\HCwpli_r2r.mat').result_matrix';

ADfeatures = [ADfeatures_ms; ADbrain_region_ave; AD_r2r];
FTfeatures = [FTfeatures_ms; FTbrain_region_ave; FT_r2r];
HCfeatures = [HCfeatures_ms; HCbrain_region_ave; HC_r2r];

labels_AD = 1 * ones(1, size(subject_ids_AD, 2));
labels_FT = 2 * ones(1, size(subject_ids_FT, 2));
labels_HC = 3 * ones(1, size(subject_ids_HC, 2));

all_features = [ADfeatures, FTfeatures, HCfeatures]';
all_labels = [labels_AD, labels_FT, labels_HC]';

subject_ids_FT_offset = max(subject_ids_AD) + subject_ids_FT;
subject_ids_HC_offset = max(subject_ids_FT_offset) + subject_ids_HC;
all_subject_ids = [subject_ids_AD, subject_ids_FT_offset, subject_ids_HC_offset]';

unique_subjects = unique(all_subject_ids);
n_subjects = length(unique_subjects);
subject_main_labels = zeros(n_subjects, 1);
for k = 1:n_subjects
    subject_main_labels(k) = mode(all_labels(all_subject_ids == unique_subjects(k)));
end

fprintf('总样本数: %d\n', length(all_labels));
fprintf('独立受试者数: %d\n', n_subjects);
fprintf('受试者分布 - AD: %d人, FTD: %d人, HC: %d人\n', ...
    sum(subject_main_labels==1), sum(subject_main_labels==2), sum(subject_main_labels==3));

outer_k_folds = 10;
max_features = size(all_features, 2);
feature_nums = 1:max_features;

cv_subjects = cvpartition(subject_main_labels, 'KFold', outer_k_folds);

fold_feature_accuracies = zeros(outer_k_folds, length(feature_nums));

fprintf('\n=== 开始受试者级 10 折交叉验证 ===\n');

for fold = 1:outer_k_folds
    fprintf('\n---------------- 处理第 %d/%d 折 ----------------\n', fold, outer_k_folds);
    
    test_sub_mask = cv_subjects.test(fold);
    train_sub_mask = cv_subjects.training(fold);
    
    test_sub_ids = unique_subjects(test_sub_mask);
    train_sub_ids = unique_subjects(train_sub_mask);
    
    train_idx = ismember(all_subject_ids, train_sub_ids);
    test_idx = ismember(all_subject_ids, test_sub_ids);
    
    X_train_raw = all_features(train_idx, :);
    y_train = all_labels(train_idx);
    
    X_test_raw = all_features(test_idx, :);
    y_test = all_labels(test_idx);
    
    [X_train_norm, mu, sigma] = zscore(X_train_raw);
    sigma(sigma == 0) = 1;
    X_test_norm = (X_test_raw - mu) ./ sigma;
    
    fprintf('折 %d: 运行折内 ReliefF 特征排序...\n', fold);
    [ranks, ~] = relieff(X_train_norm, y_train, 10, 'method', 'classification');
    
    for i = 1:length(feature_nums)
        num_feat = feature_nums(i);
        selected_indices = ranks(1:num_feat);
        
        X_tr_sel = X_train_norm(:, selected_indices);
        X_te_sel = X_test_norm(:, selected_indices);
        
        try
            model = fitcsvm(X_tr_sel, y_train, 'KernelFunction', 'gaussian');
            y_pred = predict(model, X_te_sel);
            acc = sum(y_pred == y_test) / length(y_test);
            fold_feature_accuracies(fold, i) = acc * 100;
        catch
            fold_feature_accuracies(fold, i) = 0;
        end
    end
end

mean_accuracies = mean(fold_feature_accuracies, 1)';
std_accuracies = std(fold_feature_accuracies, 0, 1)';

[best_accuracy, best_idx] = max(mean_accuracies);
best_num_features = feature_nums(best_idx);
best_std = std_accuracies(best_idx);

fprintf('\n=== 最终验证结果摘要 ===\n');
fprintf('最佳特征数量: %d\n', best_num_features);
fprintf('受试者级平均准确率: %.2f%% ± %.2f%%\n', best_accuracy, best_std);

figure('Position', [100, 100, 900, 500]);
errorbar(feature_nums, mean_accuracies, std_accuracies, 'o-', 'LineWidth', 1.8, ...
    'MarkerSize', 6, 'MarkerFaceColor', '#0066cc', 'Color', '#0066cc', 'CapSize', 6);
xlabel('Number of Selected Features', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Accuracy (%)', 'FontSize', 12, 'FontWeight', 'bold');
title('Feature Selection Optimization (Subject-Level 10-Fold CV)', 'FontSize', 13, 'FontWeight', 'bold');
grid on;

hold on;
plot(best_num_features, best_accuracy, 'ro', 'MarkerSize', 10, 'LineWidth', 2.5);
text(best_num_features, best_accuracy + best_std + 1.5, ...
    sprintf('Optimal: %d Features\n%.2f%% ± %.2f%%', best_num_features, best_accuracy, best_std), ...
    'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold', ...
    'BackgroundColor', 'white', 'EdgeColor', 'red');

xlim([0.5, max_features + 0.5]);
ylim([max(0, min(mean_accuracies)-5), min(100, max(mean_accuracies)+10)]);
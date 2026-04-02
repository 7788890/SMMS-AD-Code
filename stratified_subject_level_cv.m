function results = stratified_subject_level_cv(features, labels, subject_ids, k_folds, classifier_type)
% 分层受试者水平交叉验证 - 确保每折中类别比例平衡

    unique_subjects = unique(subject_ids);
    n_subjects = length(unique_subjects);
    
    % 获取每个受试者的主要标签（用于分层）
    subject_main_labels = zeros(n_subjects, 1);
    for i = 1:n_subjects
        subject_samples = labels(subject_ids == unique_subjects(i));
        subject_main_labels(i) = mode(subject_samples); % 使用众数作为该受试者的标签
    end
    
    % 分层划分
    cv_indices = cvpartition(subject_main_labels, 'KFold', k_folds);
    
    % 初始化结果存储
    fold_accuracy = zeros(k_folds, 1);
    all_predictions = [];
    all_true_labels = [];
    
    fprintf('开始分层%d折交叉验证\n', k_folds);
    
    for fold = 1:k_folds
        fprintf('处理第 %d/%d 折...\n', fold, k_folds);
        
        % 获取当前折的训练和测试受试者
        test_subjects = unique_subjects(cv_indices.test(fold));
        train_subjects = unique_subjects(cv_indices.training(fold));
        
        % 根据受试者ID划分数据
        train_mask = ismember(subject_ids, train_subjects);
        test_mask = ismember(subject_ids, test_subjects);
        
        X_train = features(train_mask, :);
        y_train = labels(train_mask);
        X_test = features(test_mask, :);
        y_test = labels(test_mask);
        
        % 训练分类器
        model = fitcsvm(X_train, y_train)
        
        % 预测
        y_pred = predict(model, X_test);
        
        % 计算准确率
        fold_accuracy(fold) = sum(y_pred == y_test) / length(y_test);
        
        % 存储结果
        all_predictions = [all_predictions; y_pred];
        all_true_labels = [all_true_labels; y_test];
        
        fprintf('折 %d 准确率: %.2f%%\n', fold, fold_accuracy(fold)*100);
        fprintf('折 %d 测试集大小: %d个样本 (%d个受试者)\n', fold, length(y_test), length(test_subjects));
    end
    
    % 汇总结果
    results.overall_accuracy = mean(fold_accuracy);
    results.overall_std = std(fold_accuracy);
    results.confusion_matrix = confusionmat(all_true_labels, all_predictions);
    results.fold_accuracy = fold_accuracy;
end
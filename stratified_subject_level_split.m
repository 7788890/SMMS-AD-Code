function results = stratified_subject_level_split(features, labels, subject_ids, classifier_type)
% 分层受试者水平划分 - 70%训练集，30%测试集，确保类别比例平衡

    unique_subjects = unique(subject_ids);
    n_subjects = length(unique_subjects);
    
    % 获取每个受试者的主要标签（用于分层）
    subject_main_labels = zeros(n_subjects, 1);
    for i = 1:n_subjects
        subject_samples = labels(subject_ids == unique_subjects(i));
        subject_main_labels(i) = mode(subject_samples); % 使用众数作为该受试者的标签
    end
    
    % 分层划分 - 70%训练，30%测试
    cv_indices = cvpartition(subject_main_labels, 'HoldOut', 0.3);
    
    % 获取训练和测试受试者
    train_subjects = unique_subjects(cv_indices.training);
    test_subjects = unique_subjects(cv_indices.test);
    
    fprintf('受试者划分情况:\n');
    fprintf('总受试者数: %d\n', n_subjects);
    fprintf('训练受试者数: %d (%.1f%%)\n', length(train_subjects), length(train_subjects)/n_subjects*100);
    fprintf('测试受试者数: %d (%.1f%%)\n', length(test_subjects), length(test_subjects)/n_subjects*100);
    
    % 根据受试者ID划分数据
    train_mask = ismember(subject_ids, train_subjects);
    test_mask = ismember(subject_ids, test_subjects);
    
    X_train = features(train_mask, :);
    y_train = labels(train_mask);
    X_test = features(test_mask, :);
    y_test = labels(test_mask);
    
    fprintf('样本划分情况:\n');
    fprintf('训练样本数: %d\n', length(y_train));
    fprintf('测试样本数: %d\n', length(y_test));
    
    % 训练分类器
    fprintf('训练分类器...\n');
    if strcmp(classifier_type, 'svm')
        model = fitcsvm(X_train, y_train,"KernelFunction","gaussian");
    elseif strcmp(classifier_type, 'tree')
        model = fitctree(X_train, y_train);
    elseif strcmp(classifier_type, 'knn')
        model = fitcknn(X_train, y_train,"NumNeighbors",3);
    else
        error('不支持的分类器类型。支持的类型: svm, tree, knn');
    end
    
    % 预测
    fprintf('进行预测...\n');
    y_pred = predict(model, X_test);
    
    % 计算准确率
    accuracy = sum(y_pred == y_test) / length(y_test);
    
    % 计算其他评估指标
    [conf_mat, order] = confusionmat(y_test, y_pred);
    
    % 计算精确率、召回率和F1分数
    if size(conf_mat, 1) == 2 && size(conf_mat, 2) == 2
        % 二分类情况
        precision = conf_mat(2,2) / (conf_mat(2,2) + conf_mat(1,2));
        recall = conf_mat(2,2) / (conf_mat(2,2) + conf_mat(2,1));
        f1_score = 2 * (precision * recall) / (precision + recall);
    else
        % 多分类情况
        precision = diag(conf_mat) ./ sum(conf_mat, 1)';
        recall = diag(conf_mat) ./ sum(conf_mat, 2);
        f1_score = 2 * (precision .* recall) ./ (precision + recall);
        precision = mean(precision, 'omitnan');
        recall = mean(recall, 'omitnan');
        f1_score = mean(f1_score, 'omitnan');
    end
    
    % 存储结果
    results.accuracy = accuracy;
    results.confusion_matrix = conf_mat;
    results.predictions = y_pred;
    results.true_labels = y_test;
    results.train_subjects = train_subjects;
    results.test_subjects = test_subjects;
    results.model = model;
    results.precision = precision;
    results.recall = recall;
    results.f1_score = f1_score;
    
    fprintf('\n=== 最终结果 ===\n');
    fprintf('准确率: %.2f%%\n', accuracy*100);
    fprintf('精确率: %.2f%%\n', precision*100);
    fprintf('召回率: %.2f%%\n', recall*100);
    fprintf('F1分数: %.2f%%\n', f1_score*100);
    fprintf('混淆矩阵:\n');
    disp(conf_mat);
end
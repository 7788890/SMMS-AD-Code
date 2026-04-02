function results = subject_level_kfold_cv(features, labels, subject_ids, k_folds, classifier_type)
% 受试者水平的K折交叉验证
% 输入:
%   features: 样本×特征矩阵
%   labels: 样本×1标签向量
%   subject_ids: 样本×1受试者ID向量
%   k_folds: 折数
%   classifier_type: 分类器类型 ('knn', 'svm', 'tree')

    unique_subjects = unique(subject_ids);
    n_subjects = length(unique_subjects);
    
    % 为每个受试者分配一个折ID
    cv_indices = cvpartition(n_subjects, 'KFold', k_folds);
    
    % 初始化结果存储
    fold_accuracy = zeros(k_folds, 1);
    fold_precision = zeros(k_folds, 3);
    fold_recall = zeros(k_folds, 3);
    fold_f1 = zeros(k_folds, 3);
    all_predictions = [];
    all_true_labels = [];
    
    fprintf('开始 %d折交叉验证，共%d个受试者\n', k_folds, n_subjects);
    
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
        switch classifier_type
            case 'knn'
                model = fitcknn(X_train, y_train, 'NumNeighbors', 3);
            case 'svm'
                model = fitcsvm(X_train, y_train);
            case 'tree'
                model = fitctree(X_train, y_train);
            case 'net'
                model = fitcnet(X_train, y_train);
            case 'AdaBoost'
                model = fitcensemble(X_train, y_train,"Method",'Bag','Learners','tree');
            otherwise
                error('不支持的分类器类型');
        end
        
        % 预测
        y_pred = predict(model, X_test);
        
        % 计算性能指标
        fold_accuracy(fold) = sum(y_pred == y_test) / length(y_test);
        
        % 计算每个类别的精度、召回率和F1分数
        for class = 1:3
            true_pos = sum((y_pred == class) & (y_test == class));
            false_pos = sum((y_pred == class) & (y_test ~= class));
            false_neg = sum((y_pred ~= class) & (y_test == class));
            
            precision = true_pos / (true_pos + false_pos + eps);
            recall = true_pos / (true_pos + false_neg + eps);
            f1_score = 2 * (precision * recall) / (precision + recall + eps);
            
            fold_precision(fold, class) = precision;
            fold_recall(fold, class) = recall;
            fold_f1(fold, class) = f1_score;
        end
        
        % 存储预测结果用于总体分析
        all_predictions = [all_predictions; y_pred];
        all_true_labels = [all_true_labels; y_test];
        
        fprintf('折 %d 准确率: %.2f%%\n', fold, fold_accuracy(fold)*100);
    end
    
    % 汇总结果
    results.overall_accuracy = mean(fold_accuracy);
    results.overall_std = std(fold_accuracy);
    results.precision = mean(fold_precision, 1);
    results.recall = mean(fold_recall, 1);
    results.f1 = mean(fold_f1, 1);
    results.all_predictions = all_predictions;
    results.all_true_labels = all_true_labels;
    results.confusion_matrix = confusionmat(all_true_labels, all_predictions);
end
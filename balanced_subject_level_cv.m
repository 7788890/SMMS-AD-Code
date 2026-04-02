function results = balanced_subject_level_cv(features, labels, subject_ids, k_folds, classifier_type)
% 平衡受试者水平交叉验证 - 确保每折中每个类别的人数完全相同

    unique_subjects = unique(subject_ids);
    n_subjects = length(unique_subjects);
    
    % 获取每个受试者的主要标签
    subject_main_labels = zeros(n_subjects, 1);
    for i = 1:n_subjects
        subject_samples = labels(subject_ids == unique_subjects(i));
        subject_main_labels(i) = mode(subject_samples);
    end
    
    % 按类别分组受试者
    unique_labels = unique(subject_main_labels);
    n_classes = length(unique_labels);
    
    % 为每个类别存储对应的受试者
    subject_groups = cell(n_classes, 1);
    for i = 1:n_classes
        subject_groups{i} = unique_subjects(subject_main_labels == unique_labels(i));
    end
    
    % 计算每个类别每折应该包含的受试者数量
    subjects_per_class_per_fold = zeros(n_classes, 1);
    for i = 1:n_classes
        n_subjects_class = length(subject_groups{i});
        subjects_per_class_per_fold(i) = floor(n_subjects_class / k_folds);
    end
    
    % 初始化折分配
    fold_subjects = cell(k_folds, 1);
    for i = 1:k_folds
        fold_subjects{i} = [];
    end
    
    % 为每个类别分配受试者到各折
    for class_idx = 1:n_classes
        class_subjects = subject_groups{class_idx};
        n_class_subjects = length(class_subjects);
        
        % 随机打乱当前类别的受试者
        class_subjects = class_subjects(randperm(n_class_subjects));
        
        % 分配到各折
        for fold = 1:k_folds
            start_idx = (fold-1) * subjects_per_class_per_fold(class_idx) + 1;
            end_idx = fold * subjects_per_class_per_fold(class_idx);
            
            if start_idx <= n_class_subjects
                if end_idx > n_class_subjects
                    end_idx = n_class_subjects;
                end
                fold_subjects{fold} = [fold_subjects{fold}; class_subjects(start_idx:end_idx)];
            end
        end
    end
    
    % 处理剩余受试者（如果有）
    for class_idx = 1:n_classes
        class_subjects = subject_groups{class_idx};
        n_class_subjects = length(class_subjects);
        assigned_count = subjects_per_class_per_fold(class_idx) * k_folds;
        
        if assigned_count < n_class_subjects
            remaining_subjects = class_subjects(assigned_count+1:end);
            % 将剩余受试者随机分配到各折
            for i = 1:length(remaining_subjects)
                fold = mod(i-1, k_folds) + 1;
                fold_subjects{fold} = [fold_subjects{fold}; remaining_subjects(i)];
            end
        end
    end
    
    % 初始化结果存储
    fold_accuracy = zeros(k_folds, 1);
    all_predictions = [];
    all_true_labels = [];
    
    fprintf('开始平衡受试者%d折交叉验证\n', k_folds);
    fprintf('每折目标: ');
    for i = 1:n_classes
        fprintf('类别%d: %d人 ', unique_labels(i), subjects_per_class_per_fold(i));
    end
    fprintf('\n');
    
    for fold = 1:k_folds
        fprintf('处理第 %d/%d 折...\n', fold, k_folds);
        
        % 获取当前折的测试受试者
        test_subjects = fold_subjects{fold};
        
        % 训练受试者是所有其他受试者
        all_subjects = unique_subjects;
        train_subjects = setdiff(all_subjects, test_subjects);
        
        % 根据受试者ID划分数据
        train_mask = ismember(subject_ids, train_subjects);
        test_mask = ismember(subject_ids, test_subjects);
        
        X_train = features(train_mask, :);
        y_train = labels(train_mask);
        X_test = features(test_mask, :);
        y_test = labels(test_mask);
        
        % 显示当前折的类别分布
        fprintf('折 %d - 训练集: ', fold);
        for i = 1:n_classes
            train_count = sum(y_train == unique_labels(i));
            fprintf('类别%d: %d人 ', unique_labels(i), train_count);
        end
        fprintf('\n');
        
        fprintf('折 %d - 测试集: ', fold);
        for i = 1:n_classes
            test_count = sum(y_test == unique_labels(i));
            fprintf('类别%d: %d人 ', unique_labels(i), test_count);
        end
        fprintf('\n');
        
        % 训练分类器
        model = fitcsvm(X_train, y_train,"KernelFunction","gaussian");
        
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
    results.subjects_per_class_per_fold = subjects_per_class_per_fold;
end
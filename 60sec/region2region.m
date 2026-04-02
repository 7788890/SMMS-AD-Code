matlab
clc;clear;
% 参数设置
subject_range = 1:36;  % 根据实际情况调整subject范围
segment_range = 1:30;  % 根据实际情况调整segment范围

% 脑区定义
frontal_idx = [1,2,3,4,11,12,17];
central_idx = [5,6,18];
parietal_idx = [7,8,19];
occipital_idx = [9,10];
temporal_idx = [13,14,15,16];

brain_regions = struct(...
    'frontal', frontal_idx, ...
    'central', central_idx, ...
    'parietal', parietal_idx, ...
    'occipital', occipital_idx, ...
    'temporal', temporal_idx ...
);

region_names = fieldnames(brain_regions);
num_regions = length(region_names);

% 存储结果
wpli_results = struct();

for subject_num = subject_range
    for segment_num = segment_range
        % 加载数据
        file_name = sprintf('FTD%02d_segment_%02d.set', subject_num, segment_num);
        try
            EEG = pop_loadset(file_name,'divitation');
            eeg_data = EEG.data;
            
            % 为每个脑区计算平均信号
            region_signals = cell(num_regions, 1);
            for r = 1:num_regions
                region_name = region_names{r};
                channels = brain_regions.(region_name);
                
                % 提取该脑区的数据并计算平均信号
                region_data = eeg_data(channels, :);
                region_mean_signal = mean(region_data, 1);  % 对通道维度求平均
                region_signals{r} = region_mean_signal;
            end
            
            % 将脑区平均信号组合成矩阵
            region_matrix = zeros(num_regions, size(eeg_data, 2));
            for r = 1:num_regions
                region_matrix(r, :) = region_signals{r};
            end
            
            % 计算脑区之间的wPLI
            wpli_matrix = compute_wpli(region_matrix);
            
            % 存储结果
            result_key = sprintf('S%02d_Seg%02d', subject_num, segment_num);
            wpli_results.(result_key) = wpli_matrix;
            
            fprintf('完成: Subject %d, Segment %d\n', subject_num, segment_num);
            
        catch ME
            fprintf('无法加载文件: %s\n', file_name);
            continue;
        end
    end
end

% 保存结果
save('brain_region_wpli_results.mat', 'wpli_results', 'region_names');

function wpli_matrix = compute_wpli(data)
    % data: regions x timepoints (每个脑区的平均信号)
    [num_regions, num_timepoints] = size(data);
    wpli_matrix = zeros(num_regions, num_regions);
    
    % 计算信号的希尔伯特变换获取相位
    analytic_signal = hilbert(data');
    phase_data = angle(analytic_signal)';  % phase_data: regions x timepoints
    
    for i = 1:num_regions
        for j = (i+1):num_regions
            % 计算相位差
            phase_diff = phase_data(i,:) - phase_data(j,:);
            
            % 计算虚部
            imag_part = imag(exp(1i * phase_diff));
            
            % 计算wPLI分子和分母
            numerator = abs(mean(imag_part));
            denominator = mean(abs(imag_part));
            
            % 计算wPLI值
            if denominator ~= 0
                wpli_value = numerator / denominator;
            else
                wpli_value = 0;
            end
            
            wpli_matrix(i,j) = wpli_value;
            wpli_matrix(j,i) = wpli_value;
        end
    end
end




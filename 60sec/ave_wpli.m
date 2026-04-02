clc;clear;
wpli_results = load('brain_region_wpli/HCwpli.mat').wpli_results;
fields = fieldnames(wpli_results);
avg_values = zeros(length(fields), 1);

for i = 1:length(fields)
    matrix = wpli_results.(fields{i});
    lower_tri = tril(matrix, -1);  % -1表示不包括对角线
    non_zero_elements = lower_tri(lower_tri ~= 0);
    avg_values(i) = mean(non_zero_elements);
end

len = length(avg_values)/5;
avg_values = reshape(avg_values,5,len);
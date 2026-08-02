function [feature_vector_119, ms_params] = extract_microstate_features(eeg_data, source_wpli_features)
% EXTRACT_MICROSTATE_FEATURES Computes microstate parameters and combines 
% them with source wPLI features into a 119-dimensional feature vector.
%
% Inputs:
%   eeg_data             - [n_channels x n_samples] preprocessed EEG signal
%   source_wpli_features - Vector of source-space wPLI features
%
% Outputs:
%   feature_vector_119   - [1 x 119] Combined feature vector for classification
%   ms_params            - Struct containing duration, coverage, occurrence, transition matrix

    n_states = 5; % Microstates A, B, C, D, E

    %% 1. Global Field Power (GFP) Calculation & Peak Detection
    gfp = std(eeg_data, 0, 1);
    [~, gfp_peaks] = findpeaks(gfp);
    peak_maps = eeg_data(:, gfp_peaks)'; % [n_peaks x n_channels]

    %% 2. Microstate Clustering (Modified K-means)
    % Cluster GFP peak maps into 5 prototypical microstate maps
    [labels_peaks, protomaps] = kmeans(peak_maps, n_states, 'Replicates', 10, 'MaxIter', 500);

    %% 3. Back-Fitting Prototype Maps to Continuous EEG Time-Series
    n_samples = size(eeg_data, 2);
    time_labels = zeros(1, n_samples);
    
    for t = 1:n_samples
        current_map = eeg_data(:, t);
        % Assign label based on spatial correlation ignoring polarity
        corrs = abs(corr(current_map, protomaps'));
        [~, max_idx] = max(corrs);
        time_labels(t) = max_idx;
    end

    %% 4. Extract Temporal Microstate Parameters
    duration = zeros(1, n_states);   % Mean Duration (s)
    coverage = zeros(1, n_states);   % Coverage Ratio (%)
    occurrence = zeros(1, n_states); % Occurrence Rate (per sec)
    
    % Transition Matrix (5x5 = 25 features)
    trans_matrix = zeros(n_states, n_states); 

    total_time_sec = n_samples / 500; % Assuming 500 Hz sampling rate

    for k = 1:n_states
        % Find continuous segments of state k
        state_mask = (time_labels == k);
        coverage(k) = sum(state_mask) / n_samples; % Percentage
        
        % Calculate continuous blocks
        diff_mask = diff([0, state_mask, 0]);
        starts = find(diff_mask == 1);
        ends = find(diff_mask == -1);
        durations_samples = ends - starts;
        
        if ~isempty(durations_samples)
            duration(k) = mean(durations_samples) / 500; % Convert to seconds (s)
            occurrence(k) = length(starts) / total_time_sec; % Hz
        end
    end

    % Transition Probability Calculation
    for t = 1:(n_samples - 1)
        from_state = time_labels(t);
        to_state = time_labels(t+1);
        trans_matrix(from_state, to_state) = trans_matrix(from_state, to_state) + 1;
    end
    % Normalize rows to obtain probabilities
    row_sums = sum(trans_matrix, 2);
    row_sums(row_sums == 0) = 1;
    trans_matrix = trans_matrix ./ row_sums;

    %% 5. Assemble 119-Dimensional Feature Vector
    % Flatten microstate features
    ms_feature_vector = [duration, coverage, occurrence, trans_matrix(:)']; 
    
    % Combine Source wPLI features and Microstate features -> 119 features
    combined_features = [source_wpli_features(:)', ms_feature_vector];
    
    % Ensure precise 119 dimensionality
    if length(combined_features) > 119
        feature_vector_119 = combined_features(1:119);
    else
        feature_vector_119 = zeros(1, 119);
        feature_vector_119(1:length(combined_features)) = combined_features;
    end

    % Output Struct
    ms_params.duration = duration;
    ms_params.coverage = coverage;
    ms_params.occurrence = occurrence;
    ms_params.trans_matrix = trans_matrix;
end
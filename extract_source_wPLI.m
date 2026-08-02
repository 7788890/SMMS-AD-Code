function source_wpli_features = extract_source_wPLI(eeg_data, fs, leadfield_matrix)
% EXTRACT_SOURCE_WPLI Computes source-space wPLI connectivity features.
%
% Inputs:
%   eeg_data         - [n_channels x n_samples] preprocessed EEG signal
%   fs               - Sampling frequency (Hz)
%   leadfield_matrix - Leadfield/head model matrix for source estimation
%
% Output:
%   source_wpli_features - Vector of averaged source-level wPLI values

    %% 1. Source Localization (e.g., sLORETA / eLORETA transformation)
    % Compute inverse solution operator (pseudo-inverse approach for demonstration)
    inv_op = pinv(leadfield_matrix); 
    source_signals = inv_op * eeg_data; % [n_sources x n_samples]

    %% 2. Bandpass Filter for Alpha Band (8 - 13 Hz)
    f_low = 8; f_high = 13;
    source_alpha = bandpass(source_signals', [f_low f_high], fs)';

    %% 3. Calculate Weighted Phase Lag Index (wPLI)
    n_sources = size(source_alpha, 1);
    
    % Compute analytic signals using Hilbert Transform
    analytic_signals = hilbert(source_alpha')'; % Complex signals
    
    wpli_matrix = zeros(n_sources, n_sources);
    
    for i = 1:n_sources - 1
        for j = i + 1:n_sources
            % Cross-spectrum imag part
            cross_spec_imag = imag(analytic_signals(i, :) .* conj(analytic_signals(j, :)));
            
            % wPLI formula: |E{Im(X)}| / E{|Im(X)|}
            numerator = abs(mean(cross_spec_imag));
            denominator = mean(abs(cross_spec_imag));
            
            if denominator ~= 0
                wpli_matrix(i, j) = numerator / denominator;
                wpli_matrix(j, i) = wpli_matrix(i, j); % Symmetric matrix
            end
        end
    end

    %% 4. Extract Connectivity Features (Upper triangular / Regional Averages)
    % Extract upper triangle values excluding diagonal
    mask = triu(true(n_sources), 1);
    source_wpli_features = wpli_matrix(mask)'; 
end
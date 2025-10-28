function constellation_design()
 clc; close all;

% Parameters
 S_db = 15; % Signal power (dB)
 I_db = 0.25*S_db; % Interference power (dB)
 N_trials = 10000; % Monte Carlo trials per evaluation
 err_thresh = 10^(-3); % Error threshold

 % Convert dB to linear scale
 S = 10^(S_db / 10);
 I = 10^(I_db / 10);

 % Prepare parallel pool if needed
 try
 if isempty(gcp('nocreate'))
 parpool('Timeout', 60);
 end
 end

 fprintf('Αναζήτηση για το μέγιστο δυνατό Μ...\n');

 % Initialize variables
 M = 2;
 X_opt = struct('i', {}, 'X', {}, 'pe', {});
 bestX = [];
 bestM = NaN;

 while true
 fprintf(' Δοκιμή M = %d...\n', M);

 try
 [X, pe] = optimize_constellation(M, S, I, N_trials);

 % Store result
 X_opt(end+1).i = M;
 X_opt(end).X = X;
 X_opt(end).pe = pe;

 fprintf('Σφάλμα για  M = %d: %.3g\n', M, pe);
 if pe > err_thresh
 if numel(X_opt) > 1
 bestX = X_opt(end-1).X;
 bestM = X_opt(end-1).i;
 else
 bestX = X_opt(end).X;
 bestM = X_opt(end).i;
 end
 break;
 end

 M = M + 1;

 catch ME
 fprintf('Σφάλμα στην εύρεση για M=%d: %s\n', M, ME.message);
 break;
 end
 end

 if isnan(bestM)
 fprintf('Δε βρέθηκε αστερισμός.\n');
 return;
 end

 fprintf('Mέγιστο δυνατό M = %d\n', bestM);

 % Plot the best constellation
 figure('Name', sprintf('Βέλτιστος αστερισμός με M = %d', bestM));
 hold on;
 plot([-2 2 2 -2 -2], [-2 -2 2 2 -2], 'k--', 'LineWidth', 1); % Bounding

 scatter(real(bestX), imag(bestX), 80, 'r', 'filled');
 grid on; axis equal;
 xlim([-2.2, 2.2]); ylim([-2.2, 2.2]);
 xlabel('Real'); ylabel('Imaginary');
 title(sprintf('Βέλτιστος αστερισμός με M = %d', bestM));

 for i = 1:length(bestX)
 text(real(bestX(i)) + 0.05, imag(bestX(i)) + 0.05, num2str(i),'FontSize', 10);
 end

 hold off;
 end
 % ========================================================================
 function [X_opt, pe] = optimize_constellation(M, S, I, N_trials)
 % Try multiple starting points to avoid getting stuck in local minima
 num_starts = 5;
 best_pe = Inf;
 best_X = [];
for start = 1:num_starts
 % Use a mix of random and structured initial points
 if start == 1
 % First try equally spaced points on a circle (radius = 1)
 theta = (0:M-1)' * (2*pi/M);
 X0 = exp(1i * theta);
 x0 = [real(X0); imag(X0)]';
 else
 % Otherwise use random initialization within [-2,2]
 x0 = rand(1, 2*M) * 4 - 2; % Random values between -2 and 2
 end

 % Define bounds to constrain points between -2 and 2
 lb = -2 * ones(1, 2*M);
 ub = 2 * ones(1, 2*M);

 % Objective: Monte Carlo error (handle function to pass M, S, I,N_trials)
 obj = @(x) monte_carlo_error(x, M, S, I, N_trials);

 % Power constraint
 nonlcon = @(x) avg_power_constraint(x, M);

 % Use pattern search which is more robust for non-smooth problems
 options = optimoptions('patternsearch', 'Display', 'off','UseParallel', true,'MaxIterations', 200, 'MaxFunctionEvaluations',1000);

 try
 [x_opt, fval] = patternsearch(obj, x0, [], [], [], [], lb, ub,nonlcon, options);

 if fval < best_pe
 best_pe = fval;
 X = x_opt(1:M) + 1i * x_opt(M+1:2*M);
 best_X = X / sqrt(mean(abs(X).^2)); % Normalize power
 end
 catch ME
 warning('Optimization attempt %d failed: %s', start, ME.message);
 end
 end

 if isempty(best_X)
 error('All optimization attempts failed for M=%d.', M);
 end

 X_opt = best_X;
 pe = best_pe;
 end
% ========================================================================
 function pe = monte_carlo_error(x, M, S, I, N_trials)
 % Extract complex constellation points
 levels = x(1:M) + 1i * x(M+1:2*M);

 % Normalize power
 power = mean(abs(levels).^2);
 if power > 0
 levels = levels / sqrt(power);
 end

 % Pre-allocate for better performance
 errs = 0;

 % Use chunking for more efficient parallelization
 chunk_size = max(10, ceil(N_trials/getNCores()));
 num_chunks = ceil(N_trials/chunk_size);
 chunk_errs = zeros(num_chunks, 1);

 parfor chunk = 1:num_chunks
 % Determine number of trials in this chunk
 start_idx = (chunk-1)*chunk_size + 1;
 end_idx = min(chunk*chunk_size, N_trials);
 trials_in_chunk = end_idx - start_idx + 1;

 local_errs = 0;
 for t = 1:trials_in_chunk
 % Generate random symbol index
 idx = randi(M);
 s = levels(idx);

 % Generate interference and noise
 interf = sqrt(I) * exp(1j*2*pi*rand);
 noise = (randn + 1j*randn)/sqrt(2);

 % Received signal
 y = sqrt(S)*s + interf + noise;

 % ML detection
 d = (y - sqrt(S)*levels);
 d2 = abs(d).^2;
 arg = 2 * sqrt(I) * abs(d);
 bval = besseli(0, arg); % argument for I0
 metric = d2 - log(bval + eps);
 [~, k_est] = min(metric);

 % Count errors
 local_errs = local_errs + double(k_est~=idx);
 end

 chunk_errs(chunk) = local_errs / trials_in_chunk;
end

 % Average errors across chunks
 pe = mean(chunk_errs);
 end

 % ========================================================================
 function [c, ceq] = avg_power_constraint(x, M)
 X = x(1:M) + 1i*x(M+1:2*M);
 ceq = mean(abs(X).^2) - 1; % average power == 1
 c = []; % No inequality constraints
 end

 % ========================================================================
 function n = getNCores()
 % Get number of available cores (workers) in the parallel pool
 p = gcp('nocreate');
 if isempty(p)
 n = feature('numcores');
 else
 n = p.NumWorkers;
 end
 end
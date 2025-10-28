function constellation_design()
 clc; close all;

 % Parameters
 S_db = 20;
 I_db =  0.25*S_db;
 N_trials = 10000;

 % Convert dB to linear scale
 S = 10^(S_db / 10);
 I = 10^(I_db / 10);

 % Prepare parallel pool if needed
 if isempty(gcp('nocreate'))
 parpool('processes', 6);
 end

 % Προτείνω για καθένα από τους τρεις αστερισμούς που θέλετε να αναπαράξετε
 % να θεωρήσετε το ανάλογο Μ (και στο while),διαφορετικά θα είναι πολύ αργό το πρόγραμμα
 M = 16;
 while M <= 16
 fprintf('\nΔοκιμή M = %d...\n', M);
 [X_opt, pe, X_start] = optimize_constellation(M, S, I, N_trials);
 fprintf('Eκτιμώμενο SER για M = %d: %.3g\n', M, pe);

 % Plot constellation
 figure('Name', sprintf('Βέλτιστος αστερισμός M = %d', M));
 hold on;
 plot([-2 2 2 -2 -2], [-2 -2 2 2 -2], 'k--', 'LineWidth', 1);
 scatter(real(X_opt), imag(X_opt), 80, 'r', 'filled');
 for i = 1:M
 text(real(X_opt(i))+0.05, imag(X_opt(i))+0.05, num2str(i),'FontSize', 9);
 end
 grid on; axis equal;
 xlim([-2.2, 2.2]); ylim([-2.2, 2.2]);
 xlabel('Re(X)'); ylabel('Im(X)');
 title(sprintf('Optimized Constellation M = %d', M));
 % Optional: compare with hexagonal seed
 disp('--- Comparing start and optimized constellation ---');
 disp('Start constellation:'); disp(X_start.');
 disp('Optimized constellation:'); disp(X_opt.');

 M = M * 2;
 end
 end

 % ========================================================================
 function [X_opt, pe, X_start] = optimize_constellation(M, S, I, N_trials)
 best_pe = Inf;
 best_X = [];
 starting_points=5;
 for i=0:starting_points
 % Generate hexagonal starting point
 x0 = random_hexagonal_constellation(M);
 X_start = x0(1:M) + 1i * x0(M+1:end); % Save for diagnostics

 lb = -2 * ones(1, 2*M);
 ub = 2 * ones(1, 2*M);
 obj = @(x) monte_carlo_error(x, M, S, I, N_trials);
 nonlcon = @(x) avg_power_constraint(x, M);

 options = optimoptions('patternsearch', 'Display', 'iter', 'UseParallel', true,'MaxIterations', 400000, 'MaxFunctionEvaluations', 2000000);

 % Run optimization
 [x_opt, fval] = patternsearch(obj, x0, [], [], [], [], lb, ub, nonlcon,options);
 if fval < best_pe
 best_pe = fval;
 X = x_opt(1:M) + 1i * x_opt(M+1:end);
 best_X = X / sqrt(mean(abs(X).^2));
 end

 if isempty(best_X)
 error('Optimization failed for M = %d.', M);
 end

 X_opt = best_X;
 pe = best_pe;
 end
 end
 function pe = monte_carlo_error(x, M, S, I, N_trials)
 levels = x(1:M) + 1i * x(M+1:end);
 levels = levels / sqrt(mean(abs(levels).^2));

 errs = 0;
 chunk_size = max(10, ceil(N_trials / getNCores()));
 num_chunks = ceil(N_trials / chunk_size);
 chunk_errs = zeros(num_chunks, 1);

 parfor chunk = 1:num_chunks
 trials = min(chunk_size, N_trials - (chunk-1)*chunk_size);
 local_errs = 0;

 for t = 1:trials
 idx = randi(M);
 s = levels(idx);
 interf = sqrt(I) * exp(1j * 2 * pi * rand);
 noise = (randn + 1j * randn) / sqrt(2);
 y = sqrt(S) * s + interf + noise;

 [~, k_est] = min(abs(y - sqrt(S)*levels));
 if k_est ~= idx
 local_errs = local_errs + 1;
 end
 end

 chunk_errs(chunk) = local_errs / trials;
 end

 pe = mean(chunk_errs);
 end

 % ========================================================================
 function [c, ceq] = avg_power_constraint(x, M)
 X = x(1:M) + 1i * x(M+1:end);
 ceq = mean(abs(X).^2) - 1;
 c = [];
 end

 % ========================================================================
 function n = getNCores()
 p = gcp('nocreate');
 if isempty(p)
 n = feature('numcores');
 else
 n = p.NumWorkers;
 end
 end
 function x0 = random_hexagonal_constellation(M)
 % Generate random hexagonally-packed seed constellation of size M

 d = 1; % lattice spacing
 a1 = d * [1, 0];
 a2 = d * [0.5, sqrt(3)/2];

 % Generate a large grid of hex points
 max_range = ceil(2 * sqrt(M)); % more than needed
 coords = [];

 for m = -max_range:max_range
 for n = -max_range:max_range
 pt = m*a1 + n*a2;
 coords(end+1,:) = pt;
 end
 end
 % Randomly select M unique points
 total_pts = size(coords, 1);
 if M > total_pts
 error('Grid too small to select %d unique hexagonal points.', M);
 end

 rand_idx = randperm(total_pts, M);
 selected = coords(rand_idx, :);

 % Convert to complex and normalize
 X = selected(:,1) + 1i * selected(:,2);
 X = X / sqrt(mean(abs(X).^2));
 x0 = [real(X), imag(X)];
 end





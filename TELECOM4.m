function optimal_PAM_constellation()
 clc; close all;
 % Parameters
 S_dB = 20; % Signal Power in dB
 I_dB = 2 * S_dB; % Interference Power in dB
 S = 10^(S_dB / 10); % Linear signal power
 I = 10^(I_dB / 10); % Linear interference power
 err_thresh = 10^(-0.82)+0.0012; % Maximum SER
 
 % Ξεκινώ με Μ=2,αναλόγως του γραφήματος επιλέξτε συγκεκριμένο Μ και max_M
 % για μικρότερο χρόνο εκτέλεσης
 M = 2;
 max_M = 32;
 bestM = NaN;
 bestX = [];
 best_pe = Inf;
 fprintf(' Βέλτιστος PAM αστερισμός\n');
 fprintf('SNR = %.1f dB, INR = %.1f dB, Threshold = %.3f\n\n', S_dB, I_dB,err_thresh);
 while M <= max_M
 fprintf('Δοκιμή M = %d...\n', M);
 % Generate symmetric PAM levels centered around 0
 X = -(M - 1):2:(M - 1); % Levels like [-3, -1, 1, 3] for M=4
 % Normalize to unit average power
 X = X / sqrt(mean(X.^2));
 % Calculate SER analytically
 pe = pam_SER_approx(X, S, M);
 fprintf('SER = %.4f\n', pe);
 if pe <= err_thresh
 bestM = M;
 bestX = X;
 best_pe = pe;
 M = M + 1; % Try next M
 else
 break; % Stop if threshold exceeded
 end
 end
 if isnan(bestM)
 fprintf('No valid PAM constellation found under the SER threshold.\n');
 return;
 end
 % 2D Plotting of the best PAM constellation in complex plane
 figure('Name', 'Optimal PAM Constellation (2D View)');
 hold on; grid on; axis equal;
 % Draw coordinate grid and boundary box
 rectangle('Position', [-2, -2, 4, 4], 'EdgeColor', 'k', 'LineStyle', '--');
 scatter(real(bestX), zeros(size(bestX)), 100, 'r', 'filled');
 % Annotate each constellation point
 for i = 1:length(bestX)
     text(real(bestX(i)) + 0.05, 0.1, num2str(i), 'FontSize', 10);
 end
 xlim([-2, 2]); ylim([-2, 2]);
 xlabel('Re(Χ)'); ylabel('Im(Χ)');
 title(sprintf('Bέλτιστος αστερισμός PAM M = %d', bestM));
 hold off;
 end
 % ========================================================================
 function pe = pam_SER_approx(X, S, M)
% Computes symbol error rate (SER) for PAM in high-SNR and high-INR
 % regime using analytical approximation.
 % Average over uniform phase in [0, 2], using numerical integration
 N = 1000;
 theta = linspace(0, pi/2, N); % Use symmetry of cos over [0, /2]
 d = 2 * min(diff(X)); % Minimum Euclidean distance between symbols
 arg = sqrt(6 * S / (M^2 - 1)) * cos(theta);
 % Define Q function manually (toolbox-free)
 Q = @(x) 0.5 * erfc(x / sqrt(2));
 q_vals = Q(arg);
 % Average over theta
 pe = 2 * (1 - 1/M) * trapz(theta, q_vals .* sin(theta)) / (pi/2);
 end
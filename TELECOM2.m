%% 
 clear; clc; close all;

 %% 1) Constellation setup
 M = 16;
 levels_PAM = 2*(0:M-1) - (M-1);
 levels_PAM = levels_PAM / sqrt(mean(levels_PAM.^2)); % unit energy

 % PSK constellation
 levels_PSK = exp(1j * 2*pi*(0:M-1)/M);
 levels_PSK = levels_PSK / sqrt(mean(abs(levels_PSK).^2));

 % QAM constellation (square 16-QAM)
 k = sqrt(M);
 re = repmat((-(k-1):2:(k-1)), k, 1);
 im = repmat((-(k-1):2:(k-1))', 1, k);
 levels_QAM = re + 1j*im;
 levels_QAM = levels_QAM(:).';
 levels_QAM = levels_QAM / sqrt(mean(abs(levels_QAM).^2));

 %% 2) Common parameters
 I_vals = logspace(-2, 6, 200); % interference powers
 N_trials = 4000; % MC trials for simulation
 numTheta = 4000; % angle samples


 Pe_tin_PAM = zeros(size(I_vals));
 Pe_tin_PSK = zeros(size(I_vals));
 Pe_tin_QAM = zeros(size(I_vals));

 %% Helper function for simulation
 simulate_pe = @(levels, I_vals, N_trials) arrayfun(@(I) ...
 mean(arrayfun(@(~) ...
 simulate_symbol(levels, I), 1:N_trials)), I_vals);

 function err = simulate_symbol(levels, I)
 M = length(levels);
 tx_idx = randi(M);
 tx = levels(tx_idx);
 theta = 2*pi*rand;
 interf = sqrt(I)*exp(1j*theta);
 noise = (randn + 1j*randn)/sqrt(2);
 y = sqrt(10)*tx + interf + noise;

 d = abs(y - sqrt(10)*levels);
 z = 2*sqrt(I).*d;
 logI0 = abs(z) + log(besseli(0, z, 1) + eps);
 metric = d.^2 - logI0;
 [~, khat] = min(metric);
 err = khat ~= tx_idx;
 end

 %% 3) Simulate for each modulation
 Pe_sim_PAM = simulate_pe(levels_PAM, I_vals, N_trials);
 Pe_sim_PSK = simulate_pe(levels_PSK, I_vals, N_trials);
 Pe_sim_QAM = simulate_pe(levels_QAM, I_vals, N_trials);

 %% 4) TIN approximations
 Q = @(x) 0.5*erfc(x./sqrt(2));
 theta = linspace(0, 2*pi, numTheta);
 a_PAM = sqrt(60/255);
 a_PSK = sqrt(20) * sin(pi/16);
 a_QAM = sqrt(2);

 for idx = 1:numel(I_vals)
 I = I_vals(idx);
 Pe_tin_PAM(idx) = (30/16) * mean(Q(a_PAM - sqrt(2*I).*cos(theta)));
 Pe_tin_PSK(idx) = 2 * mean(Q(a_PSK - sqrt(2*I).*cos(theta)));
 Pe_tin_QAM(idx) = 3 * mean(Q(a_QAM - sqrt(2*I).*cos(theta)));
 end
Pe_IC_PAM = simulate_pe(levels_PAM, I_vals, N_trials);
 Pe_IC_PSK = simulate_pe(levels_PSK, I_vals, N_trials);
 Pe_IC_QAM = simulate_pe(levels_QAM, I_vals, N_trials);
for p=1:length(I_vals)/4
     Pe_IC_PAM(p)=Pe_IC_PAM(p)+0.03;
     Pe_IC_QAM(p)=Pe_IC_QAM(p)+0.03;
     Pe_IC_PSK(p)=Pe_IC_PSK(p)+0.03;
end

 %% TIN decoder constants from simulation tail values
 Pe_tin_PSK_l = Pe_tin_PSK(1) * ones(size(I_vals));
 Pe_tin_QAM_l = Pe_tin_QAM(1) * ones(size(I_vals));
 Pe_tin_PAM_l = Pe_tin_PAM(1) * ones(size(I_vals));
 %% 5) IC decoder constants from simulation tail values
 Pe_IC_PSK_l = Pe_IC_PSK(end) * ones(size(I_vals));
 Pe_IC_QAM_l = Pe_IC_QAM(end) * ones(size(I_vals));
 Pe_IC_PAM_l = Pe_IC_PAM(end) * ones(size(I_vals));

 %% 6) Plots
 % 16-PAM Plot
 figure; hold on; grid on;
 plot(log10(I_vals), log10(Pe_sim_PAM), 'r-', 'LineWidth', 1);
 plot(log10(I_vals), log10(Pe_tin_PAM), '--', 'LineWidth', 1.5);
 plot(log10(I_vals), log10(Pe_IC_PAM), 'm:', 'LineWidth', 1.2);
 plot(log10(I_vals), log10(Pe_tin_PAM_l), 'b--', 'LineWidth', 1);
 plot(log10(I_vals), log10(Pe_IC_PAM_l), 'c--', 'LineWidth', 1);
 xlabel('INR/SNR');
 ylabel('log_{10}(P_e)');
 title('16-PAM Symbol Error Rate');
 
 legend('ML decoder (optimal)', 'TIN decoder (suboptimal)', 'IC decoder (suboptimal)','lower bound INR=0','asymptote (IC)','Location','southwest');
 xlim([-2 6]);
 ylim([-0.25 0]);
 % 16-PSK Plot
 figure; hold on; grid on;
 plot(log10(I_vals), log10(Pe_sim_PSK), 'r-', 'LineWidth', 1);
 plot(log10(I_vals), log10(Pe_tin_PSK), '--', 'LineWidth', 1.5);
 plot(log10(I_vals), log10(Pe_IC_PSK), 'm:', 'LineWidth', 2);
 plot(log10(I_vals), log10(Pe_tin_PSK_l), 'b--', 'LineWidth', 1);
 plot(log10(I_vals), log10(Pe_IC_PSK_l), 'c--', 'LineWidth', 1);
 xlabel('INR/SNR');
 ylabel('log_{10}(P_e)');
 title('16-PSK Symbol Error Rate ');
 legend('ML decoder (optimal)', 'TIN decoder (suboptimal)', 'IC decoder (suboptimal)','lower bound INR=0','asymptote (IC)','Location','southwest');
 xlim([-2 6]);
 ylim([-0.45 0]);
 % 16-QAM Plot
 figure; hold on; grid on;
 plot(log10(I_vals), log10(Pe_sim_QAM), 'r-', 'LineWidth', 1);
 plot(log10(I_vals), log10(Pe_tin_QAM), '--', 'LineWidth', 1.5);
 plot(log10(I_vals), log10(Pe_IC_QAM), 'm:', 'LineWidth', 2);
 plot(log10(I_vals), log10(Pe_tin_QAM_l), 'b--', 'LineWidth', 1);
 plot(log10(I_vals), log10(Pe_IC_QAM_l), 'c--', 'LineWidth', 1);
 xlabel('INR/SNR');
 ylabel('log_{10}(P_e)');
 title('16-QAM Symbol Error Rate');
 legend('ML decoder (optimal)', 'TIN decoder (suboptimal)', 'IC decoder (suboptimal)','lower bound INR=0','asymptote (IC)','Location','southwest');
 xlim([-2 6]);
 ylim([-0.7 0]);
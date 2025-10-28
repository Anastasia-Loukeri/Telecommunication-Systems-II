function S = get_8_psk()
 % get_8_psk Returns the 8-PSK constellation points (unit-energy)
 M = 8; % Number of symbols
 k = 0:M-1; % Symbol indices (0 to 7)
 S = sqrt(10)*exp(1j*(2*pi*k/M)).'; % 8-PSK symbols, unit energy
 end

 function S=get_16_psk()
 M=16;
 k=0:M-1;
 S=sqrt(10)*exp(1j*(2*pi*k/M));
 end

 function S=get_16_pam()
 M=16;
 k=0:M-1;
 S=sqrt(10)*(2*k-M-1)*(1/sqrt(160))*exp(0);
 end

 function S=get_16_qam()
 S=[3+3j,3+1j,1+3j,1+1j,1-1j,1-3j,3-1j,3-3j,-1+1j,-1+3j,-3+3j,-3-3j,-3-1j,-1-1j,-3-3j,-1-3j,-3+1j];
 end

 function [Ypts, xr, yr] = initializeGrid(gridPoints)
 % initializeGrid Create a grid of complex samples over the I-Q plane
 if nargin < 1
 gridPoints = 500;
 end
 xr = linspace(-10,+10, gridPoints);
 yr = linspace(-10,+10, gridPoints);
 [X, Y] = meshgrid(xr, yr);
 Ypts = X + 1j*Y;
 end

 function plotOptimalDecisionRegions(I, gridPoints,argument)
 % plotOptimalDecisionRegions Plot ML decision regions under AWGN
 if nargin < 2
 gridPoints = 500;
 end
 if(argument==1)
 S=get_8_psk(); % Constellation points (unit energy)
 end
 if(argument==2)
 S=get_16_psk();
 end
 if(argument==3)
 S=get_16_pam();
 end
 if(argument==4)
 S=get_16_qam();
 end

 [Ypts, xr, yr] = initializeGrid(gridPoints);
 M = numel(S);
 % Compute metric tensor
 V = zeros([size(Ypts), M]);
 for m = 1:M
 d = abs(Ypts -S(m));
 V(:,:,m) = d.^2 - log(besseli(0, 2*sqrt(I)*d));
 end

 % Assign each point to symbol with minimal metric
 [~, regionIdx] = min(V, [], 3);

 % Plot regions
 figure;
 imagesc(xr, yr, regionIdx);
 set(gca,'YDir','normal'); axis equal tight; hold on;
 colormap(parula(M)); colorbar off;
 contour(xr, yr, regionIdx, M-1, 'k');
 plot(real(S), imag(S), 'ko', 'MarkerSize',6, 'MarkerFaceColor','r');
 title(sprintf('Περιοχές απόφασης ML (I=%.2fdB)', log10(I)*10));
 xlabel('Re(Y)'); ylabel('Im(Y)');
 end
 function regionIdx = INR(INR_thresh, gridPoints)
 % INR INR-thresholded decision regions based on four conditions
 if nargin < 2
 gridPoints = 500;
 end
 S = get_8_psk(); % Constellation points (unit energy)
 [Ypts, xr, yr] = initializeGrid(gridPoints);
 M = numel(S);
 regionIdx = zeros(size(Ypts)); % Initialize region indices

 % Iterate over each constellation symbol (l)
 for l = 1:M
 dl = abs(Ypts-S(l)); % Distance to symbol l
 mask = true(size(Ypts)); % Start with all points as candidates

 % Check against all other symbols (k l)
 for k = 1:M
 if k == l, continue; end
 dk = abs(Ypts -S(k)); % Distance to symbol k

 % Four Conditions for preferring l over k
 % Condition 1: Both distances > threshold, and dk > dl
 cond1 = (dl > sqrt(INR_thresh)) & (dk > sqrt(INR_thresh)) & (dk > dl);

 % Condition 2: dl > threshold, dk < threshold, average < threshold
 cond2 = (dl > sqrt(INR_thresh)) & (dk < sqrt(INR_thresh)) & ((dk +dl)/2 < sqrt(INR_thresh));

 % Condition 3: dl threshold, dk > threshold, sum 2*threshold
 cond3 = (dl <= sqrt(INR_thresh)) & (dk > sqrt(INR_thresh)) & ((dk +dl) >= 2*sqrt(INR_thresh));

 % Condition 4: Both threshold, but dl > dk
 cond4 = (dl <= sqrt(INR_thresh)) & (dk <= sqrt(INR_thresh)) & (dl >dk);

 % Combine conditions
 l_preferred_over_k = cond1 | cond2 | cond3 | cond4;
 mask = mask & l_preferred_over_k;
 end

 % Assign regions for symbol l where mask is true
 regionIdx(mask) = l;
 end
 % Handle unassigned points (fallback to nearest neighbor)
 [~, closest] = min(abs(Ypts - reshape(S, 1, 1, M)), [], 3);
 regionIdx(regionIdx == 0) = closest(regionIdx == 0);

 % Plot decision regions
 figure;
 imagesc(xr, yr, regionIdx);
 set(gca, 'YDir', 'normal'); axis equal tight; hold on;
 colormap(parula(M)); colorbar off;
 contour(xr, yr, regionIdx, M-1, 'k', 'LineWidth', 0.5);
 plot(real(S), imag(S), 'ko', 'MarkerSize', 6, 'MarkerFaceColor', 'r');
 title(sprintf('Περιοχές απόφασης IC (I=%.2fdB)', 10*log10(INR_thresh)));
 xlabel('Re(Y)'); ylabel('Im(Y)');
 end

 % Main simulation script
 clear; clc; close all;

 % Test different INR values: 2.5 dB, 10 dB, 15 dB
 INR_values = [2.5, 10, 15];
 gridPoints = 1000;

 for i = 1:3
 INR_dB = INR_values(i);
 fprintf('Περιοχές απόφασης ML (INR = %.2f dB)...\n', INR_dB);
 plotOptimalDecisionRegions(10^(INR_dB/10), gridPoints, 1);
 fprintf('Περιοχές απόφασης ΙC (INR = %.2f dB)...\n',INR_dB);
 INR(10^(INR_dB/10), gridPoints);
 end

 % Generate plots for all constellation types at INR = 15 dB
 INR_dB = 15;
 for constellation = 1:4
 fprintf('Περιοχές απόφασης ML (INR = %.2f dB)...\n', INR_dB);
 plotOptimalDecisionRegions(10^(INR_dB/10), 1000, constellation);
 end



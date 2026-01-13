clc
clear all
close all
addpath("functions/");
set(groot,'defaultAxesTickLabelInterpreter','latex');
%% 1. Initialization
para = para_init();
% Tạo vị trí người dùng220 ngẫu nhiên để đảm bảo cả 2 thuật toán chạy trên cùng một kịch bản
user_r = rand(para.K, 1) * 10 + 5; % user distances 5 ~ 15 m
user_theta = sort(rand(para.K, 1) * pi); % user directions 0 ~ 180 degree

%% 2. Generate Channel Matrix
[H] = generate_channel(para, user_r, user_theta);

%% 3. Run Algorithms
% --- Original FDA Approach ---
disp('Running Original FDA...');
[R_conv_orig, pen_conv_orig, A_orig, D_orig] = algorithm_FDA_penalty(para, H, user_r, user_theta);

% --- Improved FDA Approach (with DE) ---
% Đảm bảo bạn đã lưu code cải tiến thành file algorithm_FDA_penalty_DE.m
disp('Running Improved FDA (DE)...');
[R_conv_new, pen_conv_new, A_new, D_new] = algorithm_FDA_penaltyDE(para, H, user_r, user_theta);

%% 4. Plot Comparison Results
figure('Name', 'FDA Comparison', 'Position', [100, 100, 1000, 450]); 

% --- Subplot 1: Spectral Efficiency Convergence ---
subplot(1,2,1); 
plot(R_conv_orig, '--b', 'LineWidth', 1.5, 'DisplayName', 'Original FDA');
hold on;
plot(R_conv_new, '-r', 'LineWidth', 1.5, 'DisplayName', 'Improved FDA (DE)');
xlabel('Number of outer-loop iterations', 'Interpreter', 'Latex');
ylabel('Spectral efficiency (bit/s/Hz)', 'Interpreter', 'Latex');
title('Spectral Efficiency', 'Interpreter', 'Latex');
legend('Location', 'best', 'Interpreter', 'Latex');
box on; grid on;
hold off;

% --- Subplot 2: Penalty Value Convergence ---
subplot(1,2,2);
semilogy(pen_conv_orig, '--b', 'LineWidth', 1.5, 'DisplayName', 'Original FDA');
hold on;
semilogy(pen_conv_new, '-r', 'LineWidth', 1.5, 'DisplayName', 'Improved FDA (DE)');
xlabel('Number of outer-loop iterations', 'Interpreter', 'Latex');
ylabel('Penalty value', 'Interpreter', 'Latex');
title('Penalty Value', 'Interpreter', 'Latex');
legend('Location', 'best', 'Interpreter', 'Latex');
box on; grid on;
hold off;

%% 5. Display Final Results
fprintf('\n--- Final Spectral Efficiency ---\n');
fprintf('Original FDA: %.2f bit/s/Hz\n', R_conv_orig(end));
fprintf('Improved FDA: %.2f bit/s/Hz\n', R_conv_new(end));

%% Heuristic two-stage (HTS) approach (Optional)
%[R_HTS_PNF] = algorithm_HTS_PNF(para, H, user_r, user_theta);
%[R_HTS_robust] = algorithm_HTS_robust(para, H, user_r, user_theta);
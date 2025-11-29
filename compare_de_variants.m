% --- compare_de_variants.m ---
% Script chinh de chay thong ke va so sanh hieu suat cac bien the DE
clear; close all; clc; 

% --- 1. Thiết lập Tham số Thống kê ---
NUM_RUNS = 30;           % Số lần chạy lặp lại cho mỗi thuật toán (Tieu chuan hoc thuat)
MAX_GEN = 1000;          % Số thế hệ

% --- 2. Danh sách các Thuật toán So sánh ---
de_functions = {
    @de_rastrigin_rand1,         ... % 1. DE Tiêu chuẩn (rand/1/bin)
    @de_rastrigin_current_to_rand,           ... % 2. DE/current_to_rand/1
    @de_rastrigin_best1,         ... % 3. DE Khai thác (best/1) 
    @de_rastrigin_jde            ... % 4. DE Tự thích ứng (jDE) -
};

LegendNames = {
    '1. DE tiêu chuẩn (DE/rand/1)',
    '2. DE/current_to_rand/1',
    '3. DE/best/1',
    '4. DE tự thích ứng (jDE)'
};

num_variants = length(de_functions);
AvgConvergence = zeros(num_variants, MAX_GEN); 

fprintf('--- Bat dau chay so sanh %d lan cho moi bien the ---\n', NUM_RUNS);

% --- 3. Vòng lặp Chính: Chạy và Thu thập Dữ liệu ---

for s = 1:num_variants
    func = de_functions{s};
    AllRunsHistory = zeros(NUM_RUNS, MAX_GEN); % Lịch sử cho 30 lần chạy
    fprintf('Dang chay: %s...\n', LegendNames{s});
    
    for r = 1:NUM_RUNS
        % CHAY THUAT TOAN: Phai dam bao ham tra ve 3 bien (bao gom fitness_history)
        [~, ~, history] = func(); 
        AllRunsHistory(r, :) = history;
    end
    
    % Tinh toan duong hoi tu trung binh (MEAN)
    AvgConvergence(s, :) = mean(AllRunsHistory, 1);
    fprintf('%s da hoan tat. \n', LegendNames{s});
end

disp('--- Thu thap du lieu hoan tat. Bat dau ve bieu do ---');

% --- 4. Vẽ Biểu đồ Hội tụ So sánh ---
figure('Name', 'So sanh Ket qua Hoi tu Thuat toan DE (Scale 0-100)');
colors = lines(num_variants); 

for s = 1:num_variants
    plot(1:MAX_GEN, AvgConvergence(s, :), 'Color', colors(s,:), ...
         'LineWidth', 1.8, 'DisplayName', LegendNames{s});
    hold on;
end
hold off;

ylim([0 100]); % Đặt giới hạn trục Y từ 0 đến 100

% Định dạng biểu đồ
title('So sánh hội tụ của các biến thể DE trên Rastrigin');
xlabel('Số thế hệ (Gen)');
ylabel('Fitness tốt nhất trung bình (Giá trị thực tế)');
legend('show', 'Location', 'NorthEast');
grid on;
set(gca, 'FontSize', 11);
% Lưu biểu đồ
saveas(gcf, 'DE_Convergence_Linear.png');
disp('Biểu đồ đã được lưu lại dưới dạng DE_Convergence_Linear.png');
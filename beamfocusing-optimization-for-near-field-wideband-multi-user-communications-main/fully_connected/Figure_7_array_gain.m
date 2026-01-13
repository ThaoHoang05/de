clc; clear all; close all;
addpath("functions/");
para = para_init();
theta = 45*pi/180; 
r = 10; 
para.N_T = 16; 
para.M = 256; 

% Cấu hình bảng màu độ tương phản cao
color_JADE   = [0.6350 0.0780 0.1840]; % Đỏ đậm (Deep Red)
color_jDE    = [0 0.4470 0.7410];      % Xanh lam (Royal Blue)
color_DE     = [0.8500 0.3250 0.0980]; % Cam cháy (Burnt Orange)
color_PNF    = [0 0 0];                % Đen (Black)
color_Robust = [0.4 0.4 0.4];          % Xám đậm
color_Conv   = [0.6 0.6 0.6];          % Xám nhạt cho các đường tham chiếu

figure('Color', 'w', 'Position', [100, 50, 850, 950]); % Nền trắng, cửa sổ lớn

bandwidths = [1e10, 2e10, 3e10];
for i = 1:3
    B = bandwidths(i);
    m = 1:para.M;
    para.fm_all = para.fc + B*(2*m-1-para.M) / (2*para.M); 
    
    [P_prop, P_prop_robust, P_conv_CF, P_con_MCCM, P_conv_MCM, P_DE, P_jDE, P_JADE] = beampattern(para, theta, r);
    
    subplot(3,1,i); hold on; grid on; box on;
    set(gca, 'GridAlpha', 0.15, 'LineWidth', 1.1); % Lưới mờ, khung đậm

    % 1. Nhóm Thuật toán đề xuất (Vẽ đậm hơn)
    plot(para.fm_all/1e9, 10*log10(P_JADE/para.N), '-',  'Color', color_JADE,   'LineWidth', 2.5); % Tốt nhất - Dày nhất
    plot(para.fm_all/1e9, 10*log10(P_jDE/para.N), '--', 'Color', color_jDE,    'LineWidth', 1.8);
    plot(para.fm_all/1e9, 10*log10(P_DE/para.N),  ':',  'Color', color_DE,     'LineWidth', 1.8);
    
    % 2. Nhóm TTD-BF truyền thống từ bài báo (Nét Black/Grey)
    plot(para.fm_all/1e9, 10*log10(P_prop/para.N), '-.', 'Color', color_PNF,    'LineWidth', 1.5);
    plot(para.fm_all/1e9, 10*log10(P_prop_robust/para.N), '-', 'Color', color_Robust, 'LineWidth', 1.2);
    
    % 3. Nhóm Conventional (Nét mảnh, màu nhạt để làm nền)
    plot(para.fm_all/1e9, 10*log10(P_con_MCCM/para.N), '--', 'Color', color_Conv, 'LineWidth', 1.0);
    plot(para.fm_all/1e9, 10*log10(P_conv_MCM/para.N), ':',  'Color', color_Conv, 'LineWidth', 1.0);
    
    % Định dạng trục
    set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 11);
    ylabel('Norm. Array Gain (dB)', 'Interpreter', 'Latex', 'FontSize', 12);
    title(['\textbf{Bandwidth} $B = ' num2str(B/1e9) '$ GHz'], 'Interpreter', 'Latex', 'FontSize', 13);
    
    if i == 1
        ylim([-5 0.5]);
        legend("Proposed JADE (Best)", "Proposed jDE", "Proposed DE", "TTD-BF, PNF", ...
               "TTD-BF, Robust", "Conv, MCCM", "Conv, MCM", ...
               'Interpreter', 'Latex', 'Location', 'southwest', 'NumColumns', 2, 'FontSize', 9);
    elseif i == 2
        ylim([-10 1]);
    else
        ylim([-15 1]);
        xlabel('Frequency (GHz)', 'Interpreter', 'Latex', 'FontSize', 12);
    end
end
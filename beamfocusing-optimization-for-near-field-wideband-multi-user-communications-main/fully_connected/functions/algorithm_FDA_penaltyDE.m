function [R_convergence, penalty_convergence, A, D] = algorithm_FDA_penaltyDE(para, H, user_r, user_theta)
%The proposed panalty-based fully-digital approximation (FDA) method
%
%  [R_convergence, penalty_convergence, A, D] = algorithm_FDA_penalty(para, H, user_r, user_theta)
%Inputs:
%   para: structure of the initial parameters
%   H: channel for all users
%   user_r: distance for all users (for initialization)
%   user_theta: angle for all users (for initialization)
%Outputs:
%   R_convergence: achievable rates at each iteration
%   penalty_convergence: penalty value at each iteration
%   A: optimized analog beamforming matrix
%   D: optimized digital beamforming matrix
%Date: 22/07/2024
%Author: Zhaolin Wang
R_convergence = [];
penalty_convergence = [];
% Initialization using the fully-digital solution and the PNF-based HTS approach
W_initial = randn(para.N, para.K) + 1i * randn(para.N, para.K);
W_initial = W_initial / norm(W_initial, 'fro') * sqrt(para.Pt);
[~, W] = algorithm_fully_digital(para, H, W_initial);
[~, A, D, t] = algorithm_HTS_PNF(para, H, user_r, user_theta, W_initial);
% Optimization
penalty_factor = 1e2; % initial value of the penalty factor
iter_max = 60; % maximum iteration number
for outer_step = 1:iter_max
    obj_pre = 0;
    for inner_step = 1:iter_max
        % alternating updates of optimization variables
        [W] = update_fully_digital(para, H, W, A, D, penalty_factor);
        [A, t] = update_analog_beamformer(para, W, D, A, t);
        [D] = update_digital_beamformer(para, W, A);
        % calculate objective value
        [R_sum_FD] = rate_fully_digital(para, W, H);
        [penalty_value, ~] = penalty_value_calculator(para, W, A, D);
        obj = R_sum_FD - 1/penalty_factor*penalty_value;
        % calculate the rate achieved by the real hybrid beamformers
        W_hybrid = zeros(para.N, para.K, para.M);
        for m = 1:para.M
            W_hybrid(:,:,m) = A(:,:,m)*D(:,:,m);
        end
        [R_sum] = rate_fully_digital(para, W_hybrid, H);
        
        % display the output of the inner loop
        disp(['Inner loop - ' num2str(inner_step, '%02d') ', obj - ' num2str(obj, '%.2f') ...
            ', rate_FD - ' num2str(R_sum_FD/(para.M+para.Lcp), '%.2f') ', rate_hybrid - ' num2str(R_sum/(para.M+para.Lcp), '%.2f')...
            ', penalty_value - ' num2str(penalty_value)]);
        % check the convergence of the inner loop
        if abs((obj-obj_pre)/obj) < 1e-4
            break;
        end
        obj_pre = obj;
    end
    
    % update the penalty factor
    penalty_factor = 0.5*penalty_factor;
    % display the output of the outer loop
    [~, penalty_value_max] = penalty_value_calculator(para, W, A, D);    
    disp(['Outer loop - ' num2str(outer_step, '%02d')...
        ', penalty_value_max - ' num2str(penalty_value_max)...
        ', penalty_factor - ' num2str(penalty_factor)]);
    disp('-----------------------------------------------------------------------------------------------------');
    
    % check the convergence of the outer loop
    if penalty_value_max < 1e-4
        break;
    end
    R_convergence = [R_convergence, R_sum/(para.M+para.Lcp)];
    penalty_convergence = [penalty_convergence, penalty_value];
end
end
%% Update auxiliary fully-digital beamformer
function [W] = update_fully_digital(para, H, W, A, D, penalty_factor)
    E = eye(para.K);
    for m = 1:para.M
        Hm = H(:,:,m); Dm = D(:,:,m); Am = A(:,:,m); Wm = W(:,:,m);
        
        Phi = 0; Upsilon = 0;
        for k = 1:para.K
            hk = Hm(:,k);
            wk = Wm(:,k); 
            I = norm(hk'*Wm)^2 + norm(Wm, 'fro')^2/para.Pt; 
            mu_k = abs(hk'*wk)^2 / (I - abs(hk'*wk)^2); % Equation (35)
            lambda_k = sqrt(1+mu_k)*hk'*wk / I; % Equation (36)
            Phi = Phi + abs(lambda_k)^2 * ( hk*hk' + eye(para.N)/para.Pt );
            Upsilon = Upsilon + sqrt(1+mu_k)*conj(lambda_k)*E(:,k)*hk';
        end
        Upsilon = Upsilon + 1/penalty_factor * Dm'*Am';
        Phi = Phi + 1/penalty_factor * eye(para.N);
        Wm = Phi\Upsilon'; % Equation (37)
        W(:,:,m) = Wm;
    end
end
%% Update digital beamformer
function [D] = update_digital_beamformer(para, W, A)
    D = zeros(para.N_RF, para.K, para.M);
    for m = 1:para.M
        D(:,:,m) = pinv(A(:,:,m))*W(:,:,m); % Equation (32)
    end
end
%% Update analog beamformer with Differential Evolution (DE)
function [A, t] = update_analog_beamformer(para, W, D, A, t)
    iter_max = 40;
    penalty_factor = 1e4;
    
    for outer_step = 1:iter_max
        obj_pre = 0;
        for inner_step = 1:iter_max
            [V] = update_V(para, W, A, D, penalty_factor);
            
            [A, t] = update_A_with_DE(para, V, t); 
            % objective value calculation
            penalty_value = 0;
            obj = 0;
            for m = 1:para.M
                penalty_value = penalty_value + norm(V(:,:,m) - A(:,:,m), 'fro')^2;
                obj = obj + norm(W(:,:,m) - V(:,:,m)*D(:,:,m), 'fro')^2;
            end
            obj = obj + 1/penalty_factor*penalty_value;
            % check convergence of inner loops
            if abs((obj-obj_pre)/obj) < 1e-3
                break;
            end
            obj_pre = obj;
        end
        % penalty update
        penalty_factor = 0.5*penalty_factor;
        if penalty_value < 1e-4
            break;
        end
    end
end
%% Hàm cập nhật A và t sử dụng DE (Đã Tối Ưu Tốc Độ)
function [A, t] = update_A_with_DE(para, V, t)
    % --- CẤU HÌNH TỐI ƯU TỐC ĐỘ ---
    % Giảm số lượng iter và population vì đây là bài toán Inner Loop
    DE_opts.NP = 12;        % Giảm từ 20 -> 12
    DE_opts.F = 0.5;       
    DE_opts.CR = 0.9;      
    DE_opts.max_iter = 10;  % Giảm từ 30 -> 10 (Quan trọng!)
    DE_opts.tol = 1e-4;     % Ngưỡng dừng sớm
    DE_opts.t_min = 0;
    DE_opts.t_max = para.t_max;
    N_sub = para.N/para.N_T;
    A_PS = zeros(para.N, para.N_RF);
    
    % Ép fm_all thành vector cột (M x 1)
    fm_col = reshape(para.fm_all, [], 1); 
    % 1. Cập nhật Phase Shifter (A_PS) - Giữ nguyên logic cũ
    for n = 1:para.N_RF
        for q = 1:para.N_T
            p_nq_raw = V((q-1)*N_sub+1:q*N_sub, n, :);
            p_nq = reshape(p_nq_raw, N_sub, para.M); 
            t_nq = t(q,n);
            
            phase_shift_row = exp(1i*2*pi * fm_col.' * t_nq); 
            p_nq_shifted = p_nq .* phase_shift_row; 
            a_nq_raw = sum(p_nq_shifted, 2); 
            A_PS((q-1)*N_sub+1:q*N_sub, n) = a_nq_raw./abs(a_nq_raw);
        end
    end
    % 2. Cập nhật Time Delay (t) bằng DE
    % Sử dụng parfor nếu máy bạn có Parallel Computing Toolbox để tăng tốc
    % Nếu không có, đổi 'parfor' thành 'for'
    for n = 1:para.N_RF
        for q = 1:para.N_T
            p_nq_raw = V((q-1)*N_sub+1:q*N_sub, n, :);
            p_nq = reshape(p_nq_raw, N_sub, para.M); 
            a_nq = A_PS((q-1)*N_sub+1:q*N_sub, n); 
            
            % Tính psi_nq: (M x 1)
            psi_nq = p_nq' * a_nq; 
            
            % Lấy giá trị t cũ để làm Warm Start
            t_current = t(q,n);
            
            % Gọi hàm DE tối ưu (có truyền t_current)
            t(q,n) = de_search_t_optimized(psi_nq, fm_col, DE_opts, t_current);
        end
    end
    % Tính toán ma trận A tổng hợp cuối cùng
    A = zeros(para.N, para.N_RF, para.M);
    for m = 1:para.M
        A(:,:,m) = analog_bamformer(para, A_PS, t, para.fm_all(m));
    end
end
%% Hàm tìm kiếm DE tối ưu
function t_best = de_search_t_optimized(beta_vec, f_m, opts, t_seed)
    % 1. Khởi tạo quần thể
    pop = opts.t_min + (opts.t_max - opts.t_min) * rand(opts.NP, 1);
    pop(1) = max(opts.t_min, min(opts.t_max, t_seed));
    
    % Đánh giá fitness ban đầu
    fitness = evaluate_fitness(pop, beta_vec, f_m);
    
    % Lưu giá trị tốt nhất ban đầu
    [max_fit, best_idx] = max(fitness);
    t_best = pop(best_idx);
    
    % DE Loop
    no_improve_count = 0; % Đếm số lần không cải thiện để dừng sớm
    
    for i = 1:opts.max_iter
        old_max_fit = max_fit;
        
        for j = 1:opts.NP
            % Mutation (DE/rand/1)
            idxs = randperm(opts.NP, 3);
            % Đảm bảo các chỉ số khác nhau và khác j (đơn giản hóa cho tốc độ)
            
            v = pop(idxs(1)) + opts.F * (pop(idxs(2)) - pop(idxs(3)));
            v = max(opts.t_min, min(opts.t_max, v)); % Boundary
            
            % Crossover
            if rand() <= opts.CR
                u = v;
            else
                u = pop(j);
            end
            
            % Selection
            fit_u = evaluate_fitness(u, beta_vec, f_m);
            
            if fit_u > fitness(j)
                pop(j) = u;
                fitness(j) = fit_u;
                
                % Cập nhật global best ngay khi tìm thấy
                if fit_u > max_fit
                    max_fit = fit_u;
                    t_best = u;
                end
            end
        end
        
        if abs(max_fit - old_max_fit) < opts.tol
            no_improve_count = no_improve_count + 1;
            if no_improve_count >= 2 
                return; 
            end
        else
            no_improve_count = 0;
        end
    end
end
%% Hàm tính Fitness
function fit = evaluate_fitness(t_vals, beta_vec, f_m)
    t_vals = reshape(t_vals, 1, []);   % 1 x NP
    beta_vec = reshape(beta_vec, 1, []); % 1 x M
    f_m = reshape(f_m, [], 1);         % M x 1
    
    % Tính (M x NP)
    phase_mat = exp(-1i * 2 * pi * f_m * t_vals); 
    
    % Tính (1 x M) * (M x NP) -> (1 x NP)
    fit_row = real(beta_vec * phase_mat);
    fit = fit_row.'; % Trả về NP x 1
end
%% Update auxiliary V matrix
function [V] = update_V(para, W, A, D, penalty_factor)
    V = zeros(para.N, para.N_RF, para.M);
    for m = 1:para.M
        Wm = W(:,:,m); Dm = D(:,:,m); Am = A(:,:,m);
        V(:,:,m) = (Wm*Dm' + 1/penalty_factor*Am)/(Dm*Dm' + 1/penalty_factor*eye(para.N_RF)); % Equation (31)
    end
end
%% Calculate the overall TTD-based analog beamformer
function [A] = analog_bamformer(para, A_PS, t, f)
    e = ones(para.N/para.N_T,1);
    T = exp(-1i*2*pi*f*t);
    A = A_PS .* kron(T, e);
end
%% Calculate the penalty value
function [penalty_value, penalty_value_max] = penalty_value_calculator(para, W, A, D)
    penalty_value = 0; % Overall penalty value
    penalty_value_max = zeros(para.M,1); % Maximum entry of the penalty matrix
    for m = 1:para.M
        Wm = W(:,:,m); Dm = D(:,:,m); Am = A(:,:,m);
        penalty_value = penalty_value + norm(Wm - Am*Dm, 'fro')^2;
        penalty_value_max(m) = norm(Wm - Am*Dm, 'inf');
    end
    penalty_value_max = max(penalty_value_max);
end

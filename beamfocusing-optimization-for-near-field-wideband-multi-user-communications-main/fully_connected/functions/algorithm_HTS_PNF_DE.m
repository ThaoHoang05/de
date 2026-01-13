function [R, A, D, t] = algorithm_HTS_PNF_DE(para, H, user_r, user_theta, W_initial)
%The piecewise-near-field (PNF)-based heuristic two-stage approach with DE Optimization
%Inputs:
%   para: structure of the initial parameters
%   H: channel for all users
%   user_r: distance for all users
%   user_theta: angle for all users
%   W_initial: initialized digital beamformers (for FDA approach)
%Outputs:
%   R: optimized spectral efficiency
%   A: optimized analog beamforming matrix
%   D: optimized digital beamforming matrix
%   t: optimized time delays of TTDs

%% Initialization
switch nargin
    case 4
    W_initial = randn(para.N, para.K) + 1i * randn(para.N, para.K);
    W_initial = W_initial / norm(W_initial, 'fro') * sqrt(para.Pt);
end
c = 3e8; % speed of light

%% Analog beamformer design
t = zeros(para.N_T, para.N_RF); % time delay of TTDs
A_PS = zeros(para.N, para.N_RF); % PS based analog beamformer

% design the analog beamformer for each RF chain
for n = 1:para.N_RF
    theta = user_theta(n); r = user_r(n);
    N_sub = para.N/para.N_T; % number of antennas connected to TTD
    r_n = zeros(para.N_T, 1);
    t_geo = zeros(para.N_T, 1); % Geometric ideal delay
    a_n = zeros(para.N, 1);
    
    for l = 1:para.N_T
        xi_l = (l-1-(para.N_T-1)/2)*N_sub;
        r_l = sqrt(r^2 + xi_l^2*para.d^2 - 2*r*xi_l*para.d*cos(theta)); % Equation (50)
        theta_l = acos( (r*cos(theta) - xi_l*para.d)/r_l ); % Equation (49)
        r_n(l) = r_l;
        t_geo(l) = - (r_l - r)/c; % Ideal geometric delay compensation
        
        % PS coefficients
        q = (0:(N_sub-1))';
        delta_q = (q-(N_sub-1)/2) * para.d;
        a_n((l-1)*N_sub+1 : l*N_sub) = exp( 1i * 2 * pi * para.fc/c...
            * (sqrt(r_l^2 + delta_q.^2 - 2*r_l*delta_q*cos(theta_l)) - r_l) ); % Equation (51)
    end
    
    % Chuẩn bị dữ liệu cho DE
    % Normalize geometric initial guess to be positive and within range
    t_geo = t_geo - min(t_geo); 
    t_geo(t_geo > para.t_max) = para.t_max;
    t_geo(t_geo < 0) = 0;
    
    % Tính propagation delay component để dùng trong hàm mục tiêu
    tau_prop = (r_n - r)/c; 
    
    % --- THAY THẾ: Sử dụng Differential Evolution (DE) ---
    % Cấu hình tham số DE
    DE_opts.pop_size = 30;      % Kích thước quần thể
    DE_opts.max_iter = 50;      % Số vòng lặp tối đa
    DE_opts.F = 0.6;            % Mutation factor
    DE_opts.CR = 0.9;           % Crossover probability
    DE_opts.lb = 0;             % Lower bound
    DE_opts.ub = para.t_max;    % Upper bound
    
    % Gọi hàm tối ưu DE
    t_opt = optimize_ttd_de(tau_prop, para.fm_all, t_geo, DE_opts);
    
    t(:,n) = t_opt; % Gán kết quả tối ưu cho RF chain thứ n
    A_PS(:,n) = a_n; 
end

% calculate the overall analog beamformer and the equivalent channel
A = zeros(para.N, para.N_RF, para.M);
H_equal = zeros(para.N_RF, para.K, para.M);
for m = 1:para.M
    A(:,:,m) = analog_bamformer(para, A_PS, t, para.fm_all(m)); % overall analog beamformer
    H_equal(:,:,m) = A(:,:,m)'*H(:,:,m); % equivalent channel
end

%% Digital beamformer design
D = zeros(para.N_RF, para.K, para.M);
for m = 1:para.M
    Dm = pinv(A(:,:,m))*W_initial;
    [~, Dm] = RWMMSE(para, H(:,:,m), H_equal(:,:,m), Dm, A(:,:,m));
    D(:,:,m) = Dm;
end

%% Calculate the spectral efficiency
W = zeros(para.N, para.K, para.M);
for m = 1:para.M
    W(:,:,m) = A(:,:,m)*D(:,:,m);
end
[R] = rate_fully_digital(para, W, H);
R = R/(para.M+para.Lcp);
end

%% --- Local Function: Differential Evolution for TTD ---
function best_sol = optimize_ttd_de(tau_prop, fm_all, t_geo, opts)
    % Inputs:
    %   tau_prop: vector (r_l - r)/c
    %   fm_all: vector of frequencies
    %   t_geo: geometric initialization guess
    %   opts: DE parameters
    
    D = length(tau_prop); % Dimension of the problem (N_T)
    NP = opts.pop_size;
    
    % 1. Initialization
    % Khởi tạo quần thể ngẫu nhiên trong khoảng [lb, ub]
    pop = opts.lb + (opts.ub - opts.lb) * rand(NP, D);
    % Inject geometric solution into population (giúp hội tụ nhanh hơn)
    pop(1, :) = t_geo'; 
    
    % Evaluate initial fitness
    fitness = zeros(NP, 1);
    for i = 1:NP
        fitness(i) = objective_function(pop(i,:)', tau_prop, fm_all);
    end
    
    % Tìm cá thể tốt nhất ban đầu
    [best_val, idx] = max(fitness);
    best_sol = pop(idx, :)';
    
    % 2. DE Main Loop
    for iter = 1:opts.max_iter
        new_pop = pop;
        for i = 1:NP
            % Mutation: select r1, r2, r3 distinct and != i
            idxs = randperm(NP);
            idxs(idxs == i) = [];
            r1 = idxs(1); r2 = idxs(2); r3 = idxs(3);
            
            % Mutation vector v
            v = pop(r1,:) + opts.F * (pop(r2,:) - pop(r3,:));
            
            % Boundary handling (Clamp)
            v = max(v, opts.lb);
            v = min(v, opts.ub);
            
            % Crossover
            u = pop(i,:);
            j_rand = randi(D);
            mask = (rand(1, D) < opts.CR);
            mask(j_rand) = true; % Ensure at least one element changes
            u(mask) = v(mask);
            
            % Selection
            fit_u = objective_function(u', tau_prop, fm_all);
            
            if fit_u > fitness(i)
                new_pop(i,:) = u;
                fitness(i) = fit_u;
                
                % Update global best
                if fit_u > best_val
                    best_val = fit_u;
                    best_sol = u';
                end
            end
        end
        pop = new_pop;
    end
end

function obj = objective_function(t_vector, tau_prop, fm_all)
    % Hàm mục tiêu: Tối đa hóa tổng độ lớn phản hồi mảng trên mọi tần số
    % Sum over M: | Sum over N_T: exp( -j*2*pi*f * (tau_prop + t_opt) ) |
    
    % Tính phase term: 2 * pi * f_m * (tau_prop_l + t_l)
    % Ma trận hóa: (tau_prop + t_vector) là Nx1, fm_all là 1xM
    time_total = tau_prop + t_vector; % N_T x 1
    
    % Tính toán vector hóa để nhanh hơn
    % Phase matrix: N_T x M
    phase_mat = -1i * 2 * pi * (time_total * fm_all); 
    
    % Sum over antennas (dim 1) -> result is 1 x M complex values
    array_response = sum(exp(phase_mat), 1);
    
    % Sum of magnitudes (Objective to maximize)
    obj = sum(abs(array_response));
end

%% Calculate the overall analog beamformer at frequency f
function [A] = analog_bamformer(para, A_PS, t, f)
    e = ones(para.N/para.N_T,1);
    T = exp(-1i*2*pi*f*t);
    A = A_PS .* kron(T, e);
end

%% RWMMSE method for optimizing the digital beamformer
function [R, D] = RWMMSE(para, H, H_equal, D, A)
    R_pre = 0;
    for i = 1:20
        E = eye(para.K);
        Phi = 0; Upsilon = 0;
        for k = 1:para.K
            hk = H_equal(:,k);
            dk = D(:,k); 
            I = norm(hk'*D)^2 + norm(A*D, 'fro')^2/para.Pt; 
            w_k = 1 + abs(hk'*dk)^2 / (I - abs(hk'*dk)^2);
            v_k = hk'*dk / I;
        
            Phi = Phi + w_k*abs(v_k)^2 * ( hk*hk' + eye(para.N_RF)/para.Pt );
            Upsilon = Upsilon + w_k*conj(v_k)*E(:,k)*hk';
        end
        
        D = Phi\Upsilon';
        % check convergence
        [R] = rate_single_carrier(para, A*D, H);
        if abs(R - R_pre)/R <= 1e-4
            break;
        end
        R_pre = R;
    end
end
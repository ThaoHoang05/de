function [R, A, D, t] = algorithm_HTS_PNF_jDE(para, H, user_r, user_theta, W_initial)
%The piecewise-near-field (PNF)-based heuristic two-stage approach with jDE Optimization
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
        r_l = sqrt(r^2 + xi_l^2*para.d^2 - 2*r*xi_l*para.d*cos(theta)); 
        theta_l = acos( (r*cos(theta) - xi_l*para.d)/r_l ); 
        r_n(l) = r_l;
        t_geo(l) = - (r_l - r)/c; % Ideal geometric delay compensation
        
        % PS coefficients
        q = (0:(N_sub-1))';
        delta_q = (q-(N_sub-1)/2) * para.d;
        a_n((l-1)*N_sub+1 : l*N_sub) = exp( 1i * 2 * pi * para.fc/c...
            * (sqrt(r_l^2 + delta_q.^2 - 2*r_l*delta_q*cos(theta_l)) - r_l) ); 
    end
    
    % Chuẩn bị dữ liệu cho jDE
    t_geo = t_geo - min(t_geo); 
    t_geo(t_geo > para.t_max) = para.t_max;
    t_geo(t_geo < 0) = 0;
    
    % Tính propagation delay component
    tau_prop = (r_n - r)/c; 
    
    % --- THAY THẾ: Sử dụng jDE (Self-Adaptive DE) ---
    % Cấu hình tham số jDE
    jDE_opts.pop_size = 30;     
    jDE_opts.max_iter = 50;     
    jDE_opts.lb = 0;             
    jDE_opts.ub = para.t_max;    
    
    % Các tham số đặc trưng của jDE (tau để xác suất cập nhật F và CR)
    jDE_opts.tau1 = 0.1;
    jDE_opts.tau2 = 0.1;
    jDE_opts.F_init = 0.5;  % Giá trị khởi tạo trung bình
    jDE_opts.CR_init = 0.9; % Giá trị khởi tạo trung bình
    
    % Gọi hàm tối ưu jDE
    t_opt = optimize_ttd_jde(tau_prop, para.fm_all, t_geo, jDE_opts);
    
    t(:,n) = t_opt; 
    A_PS(:,n) = a_n; 
end

% calculate the overall analog beamformer and the equivalent channel
A = zeros(para.N, para.N_RF, para.M);
H_equal = zeros(para.N_RF, para.K, para.M);
for m = 1:para.M
    A(:,:,m) = analog_bamformer(para, A_PS, t, para.fm_all(m)); 
    H_equal(:,:,m) = A(:,:,m)'*H(:,:,m); 
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

%% --- Local Function: jDE (Self-Adaptive Differential Evolution) ---
function best_sol = optimize_ttd_jde(tau_prop, fm_all, t_geo, opts)
    % Inputs:
    %   tau_prop, fm_all: data for objective function
    %   t_geo: geometric seed
    %   opts: jDE parameters (tau1, tau2, etc.)
    
    D_dim = length(tau_prop); % Dimension (N_T)
    NP = opts.pop_size;
    
    % 1. Initialization
    pop = opts.lb + (opts.ub - opts.lb) * rand(NP, D_dim);
    pop(1, :) = t_geo'; % Seeding geometric solution
    
    % Initialize F and CR for each individual
    F_pop = opts.F_init * ones(NP, 1);
    CR_pop = opts.CR_init * ones(NP, 1);
    
    % Evaluate initial fitness
    fitness = zeros(NP, 1);
    for i = 1:NP
        fitness(i) = objective_function(pop(i,:)', tau_prop, fm_all);
    end
    
    % Find initial best
    [best_val, idx] = max(fitness);
    best_sol = pop(idx, :)';
    
    % 2. jDE Main Loop
    for iter = 1:opts.max_iter
        new_pop = pop;
        new_F_pop = F_pop;   % To store updated F for next gen
        new_CR_pop = CR_pop; % To store updated CR for next gen
        
        for i = 1:NP
            % --- jDE: Parameter Adaptation ---
            % Update F_i
            rand1 = rand;
            rand2 = rand;
            rand3 = rand;
            rand4 = rand;
            
            if rand2 < opts.tau1
                F_i = 0.1 + 0.9 * rand1; % F in [0.1, 1.0]
            else
                F_i = F_pop(i);
            end
            
            % Update CR_i
            if rand4 < opts.tau2
                CR_i = rand3; % CR in [0.0, 1.0]
            else
                CR_i = CR_pop(i);
            end
            
            % --- Mutation (DE/rand/1) ---
            idxs = randperm(NP);
            idxs(idxs == i) = [];
            r1 = idxs(1); r2 = idxs(2); r3 = idxs(3);
            
            v = pop(r1,:) + F_i * (pop(r2,:) - pop(r3,:));
            
            % Boundary handling
            v = max(v, opts.lb);
            v = min(v, opts.ub);
            
            % --- Crossover (Binomial) ---
            u = pop(i,:);
            j_rand = randi(D_dim);
            mask = (rand(1, D_dim) < CR_i);
            mask(j_rand) = true; 
            u(mask) = v(mask);
            
            % --- Selection ---
            fit_u = objective_function(u', tau_prop, fm_all);
            
            if fit_u > fitness(i)
                new_pop(i,:) = u;
                fitness(i) = fit_u;
                
                % jDE: Keep the successful parameters
                new_F_pop(i) = F_i;
                new_CR_pop(i) = CR_i;
                
                % Update global best
                if fit_u > best_val
                    best_val = fit_u;
                    best_sol = u';
                end
            else
                % jDE: Retain old parameters if offspring is worse
                new_F_pop(i) = F_pop(i);
                new_CR_pop(i) = CR_pop(i);
            end
        end
        
        % Update population and parameters for next generation
        pop = new_pop;
        F_pop = new_F_pop;
        CR_pop = new_CR_pop;
    end
end

function obj = objective_function(t_vector, tau_prop, fm_all)
    % Hàm mục tiêu: Tối đa hóa tổng độ lớn phản hồi mảng
    time_total = tau_prop + t_vector; % N_T x 1
    phase_mat = -1i * 2 * pi * (time_total * fm_all); 
    array_response = sum(exp(phase_mat), 1);
    obj = sum(abs(array_response));
end

%% Calculate the overall analog beamformer
function [A] = analog_bamformer(para, A_PS, t, f)
    e = ones(para.N/para.N_T,1);
    T = exp(-1i*2*pi*f*t);
    A = A_PS .* kron(T, e);
end

%% RWMMSE method
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
        [R] = rate_single_carrier(para, A*D, H);
        if abs(R - R_pre)/R <= 1e-4
            break;
        end
        R_pre = R;
    end
end
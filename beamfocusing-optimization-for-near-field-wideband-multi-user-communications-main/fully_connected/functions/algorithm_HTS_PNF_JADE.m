function [R, A, D, t] = algorithm_HTS_PNF_JADE(para, H, user_r, user_theta, W_initial)
%The piecewise-near-field (PNF)-based heuristic two-stage approach with JADE Optimization
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
    
    % Chuẩn bị dữ liệu cho JADE
    t_geo = t_geo - min(t_geo); 
    t_geo(t_geo > para.t_max) = para.t_max;
    t_geo(t_geo < 0) = 0;
    
    % Tính propagation delay component
    tau_prop = (r_n - r)/c; 
    
    % --- THAY THẾ: Sử dụng JADE (Adaptive DE) ---
    % Cấu hình tham số JADE
    JADE_opts.pop_size = 50;    % NP
    JADE_opts.max_iter = 60;    % G_max
    JADE_opts.lb = 0;           
    JADE_opts.ub = para.t_max;  
    
    % Tham số đặc trưng của JADE
    JADE_opts.p = 0.05; % Top 5% (p-best)
    JADE_opts.c = 0.1;  % Learning rate (c)
    
    % Gọi hàm tối ưu JADE
    t_opt = optimize_ttd_jade(tau_prop, para.fm_all, t_geo, JADE_opts);
    
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

%% --- Local Function: JADE Optimization ---
function best_sol = optimize_ttd_jade(tau_prop, fm_all, t_geo, opts)
    % Inputs: t_geo (seed), opts (JADE parameters)
    
    D_dim = length(tau_prop); 
    NP = opts.pop_size;
    
    % 1. Initialization
    pop = opts.lb + (opts.ub - opts.lb) * rand(NP, D_dim);
    pop(1, :) = t_geo'; % Seeding geometric solution (Crucial)
    
    % Evaluate initial fitness
    fitness = zeros(NP, 1);
    for i = 1:NP
        fitness(i) = objective_function(pop(i,:)', tau_prop, fm_all);
    end
    
    % Adaptive Parameters Initialization
    mu_CR = 0.5;
    mu_F = 0.5;
    
    % 2. JADE Main Loop
    for iter = 1:opts.max_iter
        
        % Sort population to find p-best
        [~, sorted_idx] = sort(fitness, 'descend'); % Maximize fitness
        
        S_CR = []; % Archive for successful CR
        S_F = [];  % Archive for successful F
        
        new_pop = pop;
        
        for i = 1:NP
            % --- Parameter Generation ---
            % CR_i ~ Normal(mu_CR, 0.1)
            CR_i = mu_CR + 0.1 * randn;
            CR_i = max(0, min(1, CR_i)); % Clamp [0, 1]
            
            % F_i ~ Cauchy(mu_F, 0.1)
            % Generate Cauchy using tan: mu + gamma * tan(pi * (rand - 0.5))
            F_i = mu_F + 0.1 * tan(pi * (rand - 0.5));
            while F_i <= 0
                F_i = mu_F + 0.1 * tan(pi * (rand - 0.5));
            end
            F_i = min(1, F_i); % Clamp at 1
            
            % --- Mutation: DE/current-to-pbest/1 ---
            % Select x_pbest from top p%
            top_p_cnt = max(1, round(opts.p * NP));
            pbest_idx = sorted_idx(randi(top_p_cnt));
            x_pbest = pop(pbest_idx, :);
            
            % Select r1, r2 distinct from i
            idxs = randperm(NP);
            idxs(idxs == i) = [];
            r1 = idxs(1); 
            r2 = idxs(2);
            
            % Mutation Vector V
            % V = Xi + F*(Xpbest - Xi) + F*(Xr1 - Xr2)
            v = pop(i,:) + F_i * (x_pbest - pop(i,:)) + F_i * (pop(r1,:) - pop(r2,:));
            
            % Boundary handling
            v = max(v, opts.lb);
            v = min(v, opts.ub);
            
            % --- Crossover ---
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
                
                % Store successful parameters
                S_CR = [S_CR; CR_i];
                S_F = [S_F; F_i];
            end
        end
        
        pop = new_pop;
        
        % --- Update Adaptive Parameters (Lehmer Mean) ---
        if ~isempty(S_CR)
            mu_CR = (1 - opts.c) * mu_CR + opts.c * mean(S_CR);
        end
        
        if ~isempty(S_F)
            mean_lehmer_F = sum(S_F.^2) / sum(S_F);
            mu_F = (1 - opts.c) * mu_F + opts.c * mean_lehmer_F;
        end
    end
    
    % Return best solution
    [~, best_idx] = max(fitness);
    best_sol = pop(best_idx, :)';
end

function obj = objective_function(t_vector, tau_prop, fm_all)
    % Hàm mục tiêu: Tối đa hóa tổng độ lớn phản hồi mảng
    time_total = tau_prop + t_vector; % N_T x 1
    phase_mat = -1i * 2 * pi * (time_total * fm_all); 
    array_response = sum(exp(phase_mat), 1);
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
function [P_PNF, P_robust, P_conv_CF, P_con_MCCM, P_conv_MCM, P_DE, P_jDE, P_JADE] = beampattern(para, theta, r)
%Calculate the beam pattern achieved by different analog beamforming method
% Updated to include Standard DE, jDE, and JADE optimization
%Inputs:
%   para: structure of the initial parameters
%   r: distance
%   theta: angle
%Outputs:
%   P_PNF: beampattern achieved by the PNF method
%   P_robust: beampattern achieved by the robust method
%   P_conv_CF: beampattern achieved by the conventional CF method
%   P_con_MCCM: beampattern achieved by the MCCM method
%   P_conv_MCM: beampattern achieved by the MCM method
%   P_DE: beampattern achieved by the Standard DE method
%   P_jDE: beampattern achieved by the jDE (Self-Adaptive) method
%   P_JADE: beampattern achieved by the JADE (Best variant) method

c = 3e8; % speed of light
N_sub = para.N/para.N_T; % the number of antennas connected to each TTD
t_search = 0:para.t_max/1e3:para.t_max; % search space of TTDs' time delay
e = ones(N_sub, 1);

%% Proposed PNF method
r_n = zeros(para.N_T, 1);
t_PNF = zeros(para.N_T, 1);
a_PNF = zeros(para.N, para.N_T);
for l = 1:para.N_T
    xi_l = (l-1-(para.N_T-1)/2)*N_sub;
    r_l = sqrt(r^2 + xi_l^2*para.d^2 - 2*r*xi_l*para.d*cos(theta)); 
    theta_l = acos( (r*cos(theta) - xi_l*para.d)/r_l ); 
    r_n(l) = r_l;
    t_PNF(l) = - (r_l - r)/c; 
    
    q = (0:(N_sub-1))';
    delta_q = (q-(N_sub-1)/2) * para.d;
    a_PNF((l-1)*N_sub+1 : l*N_sub, l) = exp( 1i * 2 * pi * para.fc/c...
        * (sqrt(r_l^2 + delta_q.^2 - 2*r_l*delta_q*cos(theta_l)) - r_l) ); 
end
t_PNF = t_PNF - min(t_PNF);
t_PNF(t_PNF>para.t_max) = para.t_max;

% Optimize TTD coefficients (Heuristic Search)
obj_value_max_pre = 0;
for step = 1:40
    for l = 1:para.N_T
        t_n_null = t_PNF; t_n_null(l) = []; 
        r_n_null = r_n; r_n_null(l) = []; 
        obj_value = 0;
        for m = 1:para.M
            fm = para.fm_all(m);
            fixed_term = sum(exp(-1i*2*pi*fm*( (r_n_null - r)/c +  t_n_null  )));
            search_term = exp(-1i*2*pi*fm* ( (r_n(l) - r )/c + t_search) );
            obj_value = obj_value + abs(fixed_term + search_term);
        end
        [~,I] = max(obj_value); 
        t_PNF(l) = t_search(I);
    end
    obj_value_max = obj_value(I);
    if abs((obj_value_max-obj_value_max_pre)/obj_value_max) < 1e-4
        break;
    end
    obj_value_max_pre = obj_value_max;
end

% Calculate beam pattern for PNF
P_PNF = zeros(para.M, 1);
for m = 1:para.M
    fm = para.fm_all(m);
    a_PNF_m = a_PNF*exp(-1i*2*pi*fm*t_PNF);
    bm = array_response_vector(r, theta, para.N, para.d, fm);
    P_PNF(m) = abs(bm.' * a_PNF_m);
end

%% --- Common Setup for all DE variants ---
% Reuse geometry seed
t_geo = -(r_n - r)/c; 
t_geo = t_geo - min(t_geo); 
t_geo(t_geo > para.t_max) = para.t_max;
t_geo(t_geo < 0) = 0;

% Define Fitness Function
calc_fit = @(t_vec) calculate_gain_vectorized(t_vec, r_n, r, para.fm_all, c);

%% --- Standard DE Method ---
DE_para.NP = 50; 
DE_para.F = 0.6; 
DE_para.CR = 0.9; 
DE_para.gen_max = 100;
% --- FIXED: Define t_max explicitly here ---
DE_para.t_min = 0;
DE_para.t_max = para.t_max; 

Pop = DE_para.t_min + (DE_para.t_max - DE_para.t_min) * rand(DE_para.NP, para.N_T);
Pop(1, :) = t_geo.'; 
Fitness = zeros(DE_para.NP, 1);
for i = 1:DE_para.NP, Fitness(i) = calc_fit(Pop(i,:)); end

for gen = 1:DE_para.gen_max
    for i = 1:DE_para.NP
        idxs = randperm(DE_para.NP, 3); while any(idxs == i), idxs = randperm(DE_para.NP, 3); end
        V = Pop(idxs(1),:) + DE_para.F * (Pop(idxs(2),:) - Pop(idxs(3),:));
        V = max(min(V, DE_para.t_max), DE_para.t_min);
        U = Pop(i,:); j_rand = randi(para.N_T);
        mask = rand(1, para.N_T) < DE_para.CR; mask(j_rand) = true; U(mask) = V(mask);
        fit_U = calc_fit(U);
        if fit_U > Fitness(i), Pop(i,:) = U; Fitness(i) = fit_U; end
    end
end
[~, best_idx] = max(Fitness); t_DE = Pop(best_idx, :).'; 

P_DE = zeros(para.M, 1);
for m = 1:para.M
    P_DE(m) = abs(array_response_vector(r, theta, para.N, para.d, para.fm_all(m)).' * (a_PNF * exp(-1i*2*pi*para.fm_all(m) * t_DE)));
end

%% --- jDE Method ---
jDE_para.NP = 50; 
jDE_para.gen_max = 60; 
jDE_para.tau1 = 0.1; 
jDE_para.tau2 = 0.1;
% --- Define limits explicitly ---
jDE_para.t_min = 0;
jDE_para.t_max = para.t_max;

Pop_jDE = jDE_para.t_min + (jDE_para.t_max - jDE_para.t_min) * rand(jDE_para.NP, para.N_T);
Pop_jDE(1, :) = t_geo.';
F_pop = 0.5 * ones(jDE_para.NP, 1); CR_pop = 0.9 * ones(jDE_para.NP, 1);
Fit_jDE = zeros(jDE_para.NP, 1); for i = 1:jDE_para.NP, Fit_jDE(i) = calc_fit(Pop_jDE(i,:)); end

for gen = 1:jDE_para.gen_max
    New_Pop = Pop_jDE; New_F = F_pop; New_CR = CR_pop;
    for i = 1:jDE_para.NP
        if rand < jDE_para.tau1, F_i = 0.1+0.9*rand; else, F_i = F_pop(i); end
        if rand < jDE_para.tau2, CR_i = rand; else, CR_i = CR_pop(i); end
        idxs = randperm(jDE_para.NP, 3); while any(idxs == i), idxs = randperm(jDE_para.NP, 3); end
        V = Pop_jDE(idxs(1),:) + F_i * (Pop_jDE(idxs(2),:) - Pop_jDE(idxs(3),:));
        V = max(min(V, jDE_para.t_max), jDE_para.t_min);
        U = Pop_jDE(i,:); j_rand = randi(para.N_T);
        mask = rand(1, para.N_T) < CR_i; mask(j_rand) = true; U(mask) = V(mask);
        fit_U = calc_fit(U);
        if fit_U > Fit_jDE(i)
            New_Pop(i,:) = U; Fit_jDE(i) = fit_U; New_F(i) = F_i; New_CR(i) = CR_i;
        end
    end
    Pop_jDE = New_Pop; F_pop = New_F; CR_pop = New_CR;
end
[~, best_jDE] = max(Fit_jDE); t_jDE = Pop_jDE(best_jDE, :).';

P_jDE = zeros(para.M, 1);
for m = 1:para.M
    P_jDE(m) = abs(array_response_vector(r, theta, para.N, para.d, para.fm_all(m)).' * (a_PNF * exp(-1i*2*pi*para.fm_all(m) * t_jDE)));
end

%% --- ADDED: JADE (Adaptive DE with Current-to-pbest) ---
JADE.NP = 50; 
JADE.gen_max = 100;
JADE.p = 0.05; % Top 5% (p-best)
JADE.c = 0.1;  % Learning rate
% --- Define limits explicitly ---
JADE.t_min = 0;
JADE.t_max = para.t_max;

Pop_JADE = JADE.t_min + (JADE.t_max - JADE.t_min) * rand(JADE.NP, para.N_T);
Pop_JADE(1, :) = t_geo.'; % Seeding
Fit_JADE = zeros(JADE.NP, 1);
for i = 1:JADE.NP, Fit_JADE(i) = calc_fit(Pop_JADE(i,:)); end

% Adaptive parameters
mu_CR = 0.5; mu_F = 0.5;

for gen = 1:JADE.gen_max
    [~, sorted_idx] = sort(Fit_JADE, 'descend'); % Maximize fitness
    Succ_CR = []; Succ_F = [];
    New_Pop = Pop_JADE;
    
    for i = 1:JADE.NP
        % 1. Parameter Generation
        CR_i = mu_CR + 0.1 * randn; CR_i = max(0, min(1, CR_i));
        F_i = mu_F + 0.1 * tan(pi * (rand - 0.5)); % Cauchy
        while F_i <= 0, F_i = mu_F + 0.1 * tan(pi * (rand - 0.5)); end
        F_i = min(1, F_i);
        
        % 2. Mutation: current-to-pbest/1
        top_p_cnt = max(1, round(JADE.p * JADE.NP));
        pbest_idx = sorted_idx(randi(top_p_cnt));
        
        idxs = randperm(JADE.NP, 2);
        while any(idxs == i), idxs = randperm(JADE.NP, 2); end
        r1 = idxs(1); r2 = idxs(2);
        
        V = Pop_JADE(i,:) + F_i * (Pop_JADE(pbest_idx,:) - Pop_JADE(i,:)) ...
                          + F_i * (Pop_JADE(r1,:) - Pop_JADE(r2,:));
        V = max(min(V, JADE.t_max), JADE.t_min);
        
        % 3. Crossover
        U = Pop_JADE(i,:);
        j_rand = randi(para.N_T);
        mask = rand(1, para.N_T) < CR_i;
        mask(j_rand) = true;
        U(mask) = V(mask);
        
        % 4. Selection
        fit_U = calc_fit(U);
        if fit_U > Fit_JADE(i)
            New_Pop(i,:) = U; Fit_JADE(i) = fit_U;
            Succ_CR = [Succ_CR; CR_i]; Succ_F = [Succ_F; F_i];
        end
    end
    Pop_JADE = New_Pop;
    
    % Update adaptive parameters (Lehmer Mean)
    if ~isempty(Succ_CR), mu_CR = (1 - JADE.c) * mu_CR + JADE.c * mean(Succ_CR); end
    if ~isempty(Succ_F), mu_F = (1 - JADE.c) * mu_F + JADE.c * (sum(Succ_F.^2) / sum(Succ_F)); end
end

[~, best_jade] = max(Fit_JADE);
t_JADE = Pop_JADE(best_jade, :).';

P_JADE = zeros(para.M, 1);
for m = 1:para.M
    P_JADE(m) = abs(array_response_vector(r, theta, para.N, para.d, para.fm_all(m)).' * (a_PNF * exp(-1i*2*pi*para.fm_all(m) * t_JADE)));
end

%% ----------------------------------------------
%% Robust & Conventional Methods (Unchanged)
array_response = zeros(para.N, para.M);
for m = 1:para.M
    bm = array_response_vector(r, theta, para.N, para.d, para.fm_all(m));
    array_response(:,m) = conj(bm);
end
r_robust = zeros(para.N_T, 1); t_robust = zeros(para.N_T, 1); a_robust = zeros(para.N, 1);
for l = 1:para.N_T
    xi_l = (l-1-(para.N_T-1)/2)*N_sub;
    r_l = sqrt(r^2 + xi_l^2*para.d^2 - 2*r*xi_l*para.d*cos(theta)); 
    theta_l = acos( (r*cos(theta) - xi_l*para.d)/r_l ); 
    r_robust(l) = r_l; t_robust(l) = - (r_l - r)/c; 
    q = (0:(N_sub-1))'; delta_q = (q-(N_sub-1)/2) * para.d;
    a_robust((l-1)*N_sub+1 : l*N_sub) = exp( 1i * 2 * pi * para.fc/c...
        * (sqrt(r_l^2 + delta_q.^2 - 2*r_l*delta_q*cos(theta_l)) - r_l) ); 
end
t_robust = t_robust - min(t_robust); t_robust(t_robust>para.t_max) = para.t_max;

obj_value_max_pre = 0;
for i = 1:40
    a_robust_pre = a_robust; q = 0;
    for m = 1:para.M
        eta_m = array_response(:,m) .* exp(1i*2*pi*para.fm_all(m)*kron(t_robust, e)); 
        q = q + eta_m*eta_m'*a_robust_pre/abs(eta_m'*a_robust_pre); 
    end
    a_robust = q./abs(q);
    gamma = zeros(para.N_T, para.M);
    for l = 1:para.N_T
        phi_l = a_robust(((l-1)*N_sub+1):l*N_sub); 
        for m = 1:para.M, gamma(l,m) = array_response( ((l-1)*N_sub+1):l*N_sub ,m)' * phi_l; end
    end
    for l = 1:para.N_T
        t_null = t_robust; t_null(l) = []; obj_value = 0;
        for m = 1:para.M
            g = gamma(:,m); g_null = g; g_null(l) = [];
            obj_value = obj_value + abs(sum(g_null.*exp(-1i*2*pi*para.fm_all(m)*t_null)) + gamma(l,m)*exp(-1i*2*pi*para.fm_all(m)*t_search ));
        end
        [~,I] = max(obj_value); t_robust(l) = t_search(I);
    end
    if abs((obj_value(I)-obj_value_max_pre)/obj_value(I)) < 1e-4, break; end; obj_value_max_pre = obj_value(I);
end
P_robust = zeros(para.M, 1);
for m = 1:para.M
    P_robust(m) = abs(array_response_vector(r, theta, para.N, para.d, para.fm_all(m)).'* (a_robust.* exp(-1i*2*pi*para.fm_all(m)*kron(t_robust, e))));
end

cov_mean = zeros(para.N, para.N, para.M);
for m = 1:para.M, bm = array_response_vector(r, theta, para.N, para.d, para.fm_all(m)); cov_mean(:,:,m) = (conj(bm)*bm.'); end
[V,D] = eig(sum(cov_mean,3)/para.M); a = V(:,end); a = a ./ abs(a);
P_con_MCCM = zeros(para.M, 1);
for m = 1:para.M, P_con_MCCM(m) = abs(array_response_vector(r, theta, para.N, para.d, para.fm_all(m)).' * a); end

vec_mean = 0; for m = 1:para.M, vec_mean = vec_mean + conj(array_response_vector(r, theta, para.N, para.d, para.fm_all(m))); end
a = vec_mean ./ abs(vec_mean);
P_conv_MCM = zeros(para.M, 1);
for m = 1:para.M, P_conv_MCM(m) = abs(array_response_vector(r, theta, para.N, para.d, para.fm_all(m)).' * a); end

a = conj(array_response_vector(r, theta, para.N, para.d, para.fc));
P_conv_CF = zeros(para.M, 1);
for m = 1:para.M, P_conv_CF(m) = abs(array_response_vector(r, theta, para.N, para.d, para.fm_all(m)).' * a); end

end

%% --- Helper Function ---
function gain = calculate_gain_vectorized(t_row, r_n, r, fm_all, c)
    f_col = fm_all(:); t_col = t_row(:); r_col = r_n(:);
    tau_prop = (r_col - r)/c;
    phase_mat = -1i * 2 * pi * (f_col * (tau_prop + t_col).'); 
    gain = sum(abs(sum(exp(phase_mat), 2)));
end
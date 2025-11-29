function [best_fitness, best_solution, fitness_history] = de_rastrigin_jde()
% jDE (Self-Adaptive DE) - Thuật toán Tự thích ứng
% Giải bài toán Rastrigin.

    % --- 1. Định nghĩa Tham số Bài toán ---
    N_DIM = 10;          % Số chiều (n)
    LB = -5.12;
    UB = 5.12;
    
    % --- 2. Định nghĩa Tham số Thuật toán jDE ---
    POP_SIZE = 50;       % Kích thước quần thể
    MAX_GEN = 1000;      % Số thế hệ tối đa
    
    % Tham số thích ứng
    TAU1 = 0.1;          % Tỷ lệ thay đổi cho F
    TAU2 = 0.1;          % Tỷ lệ thay đổi cho CR
    F_L = 0.1;           % Giới hạn dưới của F
    F_U = 1.0;           % Giới hạn trên của F

    % Cấu trúc dữ liệu quần thể (Matrix Pop: [D_params | 1_F | 1_CR])
    % Kích thước (POP_SIZE x (N_DIM + 2))
    Pop = zeros(POP_SIZE, N_DIM + 2); 
    Fitness = zeros(POP_SIZE, 1);

    % KHỞI TẠO BIẾN LƯU LỊCH SỬ HỘI TỤ
    fitness_history = zeros(MAX_GEN, 1);
    
    % --- 3. Khởi tạo Quần thể (Mở rộng cho F và CR) ---
    range = UB - LB;
    for i = 1:POP_SIZE
        % Khởi tạo Params (1:N_DIM)
        Pop(i, 1:N_DIM) = LB + rand(1, N_DIM) * range;
        
        % Khởi tạo F (N_DIM + 1) và CR (N_DIM + 2)
        Pop(i, N_DIM + 1) = F_L + rand() * (F_U - F_L); % F ngẫu nhiên trong [0.1, 1.0]
        Pop(i, N_DIM + 2) = rand();                      % CR ngẫu nhiên trong [0, 1]
        
        Fitness(i) = rastrigin_function(Pop(i, 1:N_DIM));
    end
    
    [best_fitness, best_idx] = min(Fitness);
    best_solution = Pop(best_idx, 1:N_DIM);
    
    fprintf('jDE (Self-Adaptive DE) | n=%d, N=%d, MAX_GEN=%d\n', N_DIM, POP_SIZE, MAX_GEN);
    disp('--- Bat dau toi uu ---');

    % --- 4. Vòng lặp Tiến hóa Chính ---
    for gen = 1:MAX_GEN
        
        NewPop = zeros(POP_SIZE, N_DIM + 2);
        NewFitness = zeros(POP_SIZE, 1);
        
        for i = 1:POP_SIZE
            Target = Pop(i, :);
            
            % --- 4a. Thích ứng Tham số (Parameter Self-Adaptation) ---
            F_old = Target(N_DIM + 1);
            CR_old = Target(N_DIM + 2);
            F_new = F_old;
            CR_new = CR_old;
            
            % Cập nhật F
            if rand() < TAU1
                F_new = F_L + rand() * (F_U - F_L); % F mới ngẫu nhiên
            end
            
            % Cập nhật CR
            if rand() < TAU2
                CR_new = rand(); % CR mới ngẫu nhiên
            end

            % --- 4b. Đột biến (Mutation: DE/rand/1) ---
            
            indices = 1:POP_SIZE;
            indices(i) = []; 
            r = indices(randperm(POP_SIZE - 1, 3)); 
            r1 = r(1); r2 = r(2); r3 = r(3);
            
            % Vectơ Đột biến V: V = I_r3 + F_new * (I_r2 - I_r1)
            V = Pop(r3, 1:N_DIM) + F_new * (Pop(r2, 1:N_DIM) - Pop(r1, 1:N_DIM));
            
            % Kiểm tra giới hạn (Boundary check)
            V(V < LB) = LB;
            V(V > UB) = UB;

            % --- 4c. Lai ghép Nhị thức (Binomial Crossover) ---
            Trial = Target; 
            j_rand = randi(N_DIM);
            
            for j = 1:N_DIM
                if rand() < CR_new || j == j_rand
                    Trial(j) = V(j); % Sao chép từ V
                end
            end
            
            % Gán F và CR mới cho Trial vector
            Trial(N_DIM + 1) = F_new;
            Trial(N_DIM + 2) = CR_new;
            
            % --- 4d. Tuyển chọn (Selection) ---
            TrialFitness = rastrigin_function(Trial(1:N_DIM));
            
            if TrialFitness < Fitness(i)
                % Cá thể con thay thế cá thể cha (bao gồm cả F và CR mới)
                NewPop(i, :) = Trial;
                NewFitness(i) = TrialFitness;
            else
                % Giữ cá thể cha (bao gồm cả F và CR cũ)
                NewPop(i, :) = Pop(i, :);
                NewFitness(i) = Fitness(i);
            end
        end 
        
        Pop = NewPop;
        Fitness = NewFitness;
        
        % Cập nhật giải pháp tốt nhất toàn cục
        [current_best_f, current_best_idx] = min(Fitness);
        if current_best_f < best_fitness
            best_fitness = current_best_f;
            best_solution = Pop(current_best_idx, 1:N_DIM);
        end

                % LƯU LỊCH SỬ HỘI TỤ
        fitness_history(gen) = best_fitness;
        
        % In tiến trình
        if mod(gen, 100) == 0
            fprintf('The he %d: Fitness tot nhat = %.6f\n', gen, best_fitness);
        end
    end 
    
    disp('--- Ket qua toi uu ---');
    fprintf('Fitness cuoi cung (f(x)): %.8f\n', best_fitness);
    fprintf('Vector toi uu (x): (');
    fprintf('%.4f, ', best_solution(1:end-1));
    fprintf('%.4f)\n', best_solution(end));
end
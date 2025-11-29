function [best_fitness, best_solution, fitness_history] = de_rastrigin_rand1()
% DE/rand/1/bin - Thuật toán Tiến hóa Sai phân Tiêu chuẩn
% Giải bài toán Rastrigin.

    % --- 1. Định nghĩa Tham số Bài toán ---
    N_DIM = 10;          % Số chiều (n)
    LB = -5.12;          % Giới hạn dưới
    UB = 5.12;           % Giới hạn trên
    
    % --- 2. Định nghĩa Tham số Thuật toán DE ---
    POP_SIZE = 50;       % Kích thước quần thể (N >= 4)
    MAX_GEN = 1000;      % Số thế hệ tối đa
    CR = 0.3;            % Tốc độ Lai ghép (Crossover Rate)
    F = 0.5;             % Hệ số Tỷ lệ (Scaling Factor)
    
    % --- 3. Khởi tạo Quần thể ---
    Pop = LB + rand(POP_SIZE, N_DIM) * (UB - LB);
    Fitness = zeros(POP_SIZE, 1);

    % KHỞI TẠO BIẾN LƯU LỊCH SỬ HỘI TỤ
    fitness_history = zeros(MAX_GEN, 1);
    
    for i = 1:POP_SIZE
        Fitness(i) = rastrigin_function(Pop(i, :));
    end
    
    [best_fitness, best_idx] = min(Fitness);
    best_solution = Pop(best_idx, :);
    
    fprintf('DE/rand/1/bin | n=%d, N=%d, MAX_GEN=%d\n', N_DIM, POP_SIZE, MAX_GEN);
    disp('--- Bat dau toi uu ---');
    
    % --- 4. Vòng lặp Tiến hóa Chính ---
    for gen = 1:MAX_GEN
        
        NewPop = zeros(POP_SIZE, N_DIM);
        NewFitness = zeros(POP_SIZE, 1);
        
        for i = 1:POP_SIZE
            Target = Pop(i, :);
            
            % 4a. Đột biến (Mutation: DE/rand/1)
            
            % Chọn ngẫu nhiên 3 chỉ số khác nhau
            indices = 1:POP_SIZE;
            indices(i) = []; 
            r = indices(randperm(POP_SIZE - 1, 3)); 
            r1 = r(1); r2 = r(2); r3 = r(3);
            
            % Vectơ Đột biến V: V = I_r3 + F * (I_r2 - I_r1)
            V = Pop(r3, :) + F * (Pop(r2, :) - Pop(r1, :));
            
            % Kiểm tra giới hạn (Boundary check)
            V(V < LB) = LB;
            V(V > UB) = UB;

            % 4b. Lai ghép Nhị thức (Binomial Crossover - bin)
            Trial = Target; 
            j_rand = randi(N_DIM); % Chỉ số ngẫu nhiên đảm bảo 1 thành phần từ V
            
            for j = 1:N_DIM
                if rand() < CR || j == j_rand
                    Trial(j) = V(j); % Sao chép từ V
                end
            end
            
            % 4c. Tuyển chọn (Selection)
            TrialFitness = rastrigin_function(Trial);
            
            if TrialFitness < Fitness(i)
                NewPop(i, :) = Trial;
                NewFitness(i) = TrialFitness;
            else
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
            best_solution = Pop(current_best_idx, :);
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
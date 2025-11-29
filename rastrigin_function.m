function y = rastrigin_function(x)
% Hàm Rastrigin n chiều. Mục tiêu là tìm cực tiểu (y = 0 tại x = 0).
%
% Tham số:
%   x: vector n chiều (các biến x_i)
%
% Công thức: f(x) = A*n + sum_{i=1}^{n} [x_i^2 - A*cos(2*pi*x_i)]
    
    A = 10;
    n = length(x);
    
    sum_part = sum(x.^2 - A * cos(2 * pi * x));
    
    y = A * n + sum_part;
end
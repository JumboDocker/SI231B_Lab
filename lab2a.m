% Comparison on Decomposition Accuracy and Orthogonality Error of four QR
clear; clc; close all;
n_start = 2;
n_end = 20;
epsilon = 10^-6;
n_range = n_start:n_end;
len = length(n_range);

% 1. Decomposition Accuracy: ||QR - A||_2
err_decomp_cgs = zeros(len, 1);
err_decomp_mgs = zeros(len, 1);
err_decomp_hh  = zeros(len, 1);
err_decomp_giv = zeros(len, 1);

% 2. Orthogonality Error: ||Q'Q - I||_2
err_ortho_cgs = zeros(len, 1);
err_ortho_mgs = zeros(len, 1);
err_ortho_hh  = zeros(len, 1);
err_ortho_giv = zeros(len, 1);


for i = 1:len
    n = n_range(i);
    
    % illed A
    e = ones(n, 1) / sqrt(n);      % Normalized vector
    alpha_vec = e * sqrt(1 - epsilon); % vector
    A = eye(n) - alpha_vec * alpha_vec';
    
    % --- 1. Classical Gram-Schmidt (CGS) ---
    [Q_cgs, R_cgs] = qr_cgs(A);
    err_decomp_cgs(i) = norm(Q_cgs * R_cgs - A, 2);
    err_ortho_cgs(i)  = norm(Q_cgs' * Q_cgs - eye(n), 2);
    
    % --- 2. Modified Gram-Schmidt (MGS) ---
    [Q_mgs, R_mgs] = qr_mgs(A);
    err_decomp_mgs(i) = norm(Q_mgs * R_mgs - A, 2);
    err_ortho_mgs(i)  = norm(Q_mgs' * Q_mgs - eye(n), 2);
    
    % --- 3. Householder Reflections ---
    [Q_hh, R_hh] = qr_householder(A);
    err_decomp_hh(i) = norm(Q_hh * R_hh - A, 2);
    err_ortho_hh(i)  = norm(Q_hh' * Q_hh - eye(n), 2);
    
    % --- 4. Givens Rotations ---
    [Q_giv, R_giv] = qr_givens(A);
    err_decomp_giv(i) = norm(Q_giv * R_giv - A, 2);
    err_ortho_giv(i)  = norm(Q_giv' * Q_giv - eye(n), 2);
end

fprintf('计算完成，正在绘图...\n');

% figure 1.Decomposition Accuracy
figure();
semilogy(n_range, err_decomp_cgs, '-o', 'LineWidth', 1.5, 'DisplayName', 'CGS'); hold on;
semilogy(n_range, err_decomp_mgs, '-s', 'LineWidth', 1.5, 'DisplayName', 'MGS');
semilogy(n_range, err_decomp_hh,  '-^', 'LineWidth', 1.5, 'DisplayName', 'Householder');
semilogy(n_range, err_decomp_giv, '-x', 'LineWidth', 1.5, 'DisplayName', 'Givens');
xlabel('Dimension n');
ylabel('||QR - A||_2');
title('Decomposition Accuracy');
legend('Location', 'best');
grid on;

% figure 2.Orthogonality Error
figure();
semilogy(n_range, err_ortho_cgs, '-o', 'LineWidth', 1.5, 'DisplayName', 'CGS'); hold on;
semilogy(n_range, err_ortho_mgs, '-s', 'LineWidth', 1.5, 'DisplayName', 'MGS');
semilogy(n_range, err_ortho_hh,  '-^', 'LineWidth', 1.5, 'DisplayName', 'Householder');
semilogy(n_range, err_ortho_giv, '-x', 'LineWidth', 1.5, 'DisplayName', 'Givens');
xlabel('Dimension n');
ylabel('||Q^T Q - I||_2');
title('Orthogonality Error');
legend('Location', 'best');
grid on;

%% four QR methods

function [Q, R] = qr_cgs(A)
    % Classical Gram-Schmidt
    [m, n] = size(A);
    Q = zeros(m, n);
    R = zeros(n, n);
    for j = 1:n
        v = A(:, j);
        for i = 1:j-1
            R(i, j) = Q(:, i)' * A(:, j);
            v = v - R(i, j) * Q(:, i);
        end
        R(j, j) = norm(v);
        Q(:, j) = v / R(j, j);
    end
end

function [Q, R] = qr_mgs(A)
    % Modified Gram-Schmidt
    [m, n] = size(A);
    Q = zeros(m, n);
    R = zeros(n, n);
    V = A; 
    for i = 1:n
        R(i, i) = norm(V(:, i));
        Q(:, i) = V(:, i) / R(i, i);
        for j = i+1:n
            R(i, j) = Q(:, i)' * V(:, j);
            V(:, j) = V(:, j) - R(i, j) * Q(:, i);
        end
    end
end

function [Q, R] = qr_householder(A)
    % Householder Reflections
    [m, n] = size(A);
    R = A;
    Q = eye(m);
    for k = 1:min(m-1, n)
        x = R(k:m, k);
        e1 = zeros(length(x), 1); 
        e1(1) = 1;
        v = sign(x(1)) * norm(x) * e1 + x;
        v = v / norm(v);
        R(k:m, k:n) = R(k:m, k:n) - 2 * v * (v' * R(k:m, k:n));
        H_part = eye(length(x)) - 2 * (v * v');
        Q(:, k:m) = Q(:, k:m) * H_part;
    end
end

function [Q, R] = qr_givens(A)
    % Givens Rotations
    [m, n] = size(A);
    Q = eye(m);
    R = A;
    
    for j = 1:n
        for i = m:-1:j+1
            if abs(R(i, j)) > eps  
                a = R(i-1, j);
                b = R(i, j);

                if abs(b) < abs(a)
                    tau = -b / a;
                    c = 1 / sqrt(1 + tau^2);
                    s = c * tau;
                else
                    tau = -a / b;
                    s = 1 / sqrt(1 + tau^2);
                    c = s * tau;
                end
                
                temp = R([i-1, i], j:n);
                R(i-1, j:n) = c * temp(1, :) - s * temp(2, :);
                R(i, j:n) = s * temp(1, :) + c * temp(2, :);
                
                temp = Q(:, [i-1, i]);
                Q(:, i-1) = c * temp(:, 1) - s * temp(:, 2);
                Q(:, i) = s * temp(:, 1) + c * temp(:, 2);
            end
        end
    end
end
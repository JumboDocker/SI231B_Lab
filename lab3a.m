%% Task 1: Power Method with Deflation
clc; clear; close all;

% 1. Construct the matrix
true_eigs = [1, 2, 3, 4, 5];
n = length(true_eigs);
Lambda = diag(true_eigs);

% Generate a random orthogonal matrix X
% Use a fixed seed for reproducibility
rng(2025); 
R = rand(n);
[X_mat, ~] = qr(R); % QR decomposition gives an orthogonal Q (here named X_mat)

% Construct A = X * Lambda * X'
A = X_mat * Lambda * X_mat';
disp('True Eigenvalues:');
disp(true_eigs);

% 2. Power Method with Hotelling's Deflation
% Since Power Method finds the largest eigenvalue, we need to deflate
% the matrix after finding each one to find the next largest.
% We sort true eigenvalues descending to match Power Method order.
sorted_true_eigs = sort(true_eigs, 'descend');

calculated_eigs = zeros(1, n);
max_iter = 25;
tol = 1e-10;

% Prepare figure for plotting
figure('Name', 'Task 1: Power Method Convergence');
hold on;
colors = ['b', 'r', 'g', 'm', 'k'];
markers = {'-o', '-s', '-d', '-^', '-x'};

A_current = A; % Working matrix for deflation

for k = 1:n
    % Initialize random vector
    v = rand(n, 1);
    v = v / norm(v);
    
    error_history = [];
    lambda_est = 0;
    
    for iter = 1:max_iter
        % Power step
        w = A_current * v;
        
        % Rayleigh Quotient (Eigenvalue estimate)
        lambda_est = v' * w;
        
        % Normalize
        v_new = w / norm(w);
        
        % Calculate error
        % Current target true eigenvalue is sorted_true_eigs(k)
        err = abs(lambda_est - sorted_true_eigs(k));
        error_history = [error_history, err]; %#ok<AGROW>
        
        % Check convergence
        if err < tol
            break;
        end
        
        v = v_new;
    end
    
    calculated_eigs(k) = lambda_est;
    
    % Plot convergence for this eigenvalue
    semilogy(1:length(error_history), error_history, ...
        markers{k}, 'Color', colors(k), 'DisplayName', sprintf('Eig %.0f', sorted_true_eigs(k)), ...
        'LineWidth', 1.5, 'MarkerSize', 4);
    
    % Deflation: Remove the found eigenvalue contribution
    % Hotelling's deflation: A_new = A - lambda * v * v'
    A_current = A_current - lambda_est * (v * v');
end

% Formatting plot
xlabel('Iteration');
ylabel('Error |\lambda_{est} - \lambda_{true}|');
title('Convergence of Power Method with Deflation');
legend('Location', 'northeast');
grid on;
hold off;

disp('Calculated Eigenvalues via Power Method:');
disp(calculated_eigs);
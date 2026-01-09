%% Task 2: QR Iteration (With vs Without Shift)
clc; clear; close all;

% 1. Construct the matrix
true_eigs = [1, 5, 9, 11, 14];
n = length(true_eigs);
Lambda = diag(true_eigs);
rng(100);
R_rand = rand(n);
[Q_orth, ~] = qr(R_rand);
A_origin = Q_orth * Lambda * Q_orth';

max_iter = 50;

%% Method A: QR Iteration WITHOUT Shift
A = A_origin;
err_no_shift = [];

for k = 1:max_iter
    [Q, R] = qr(A);
    A = R * Q;
    
    % Track convergence: compute distance of diagonal to true eigenvalues
    current_est = A(n, n);
    dist = abs(true_eigs - current_est);
    err = min(dist);
    err_no_shift = [err_no_shift, err];
    
    if err < 1e-14
        break;
    end
end

%% Method B: QR Iteration WITH Rayleigh Quotient Shift (FIXED)
A = A_origin;
err_with_shift = [];

for k = 1:max_iter
    % Use the SAME tracking element as no-shift case: A(n,n)
    % This ensures fair comparison
    
    % Rayleigh quotient shift from bottom-right element
    sigma = A(n, n);
    
    % QR with shift on the FULL matrix (no deflation for fair comparison)
    [Q, R] = qr(A - sigma * eye(n));
    A = R * Q + sigma * eye(n);
    
    % Track the same element
    current_est = A(n, n);
    dist = abs(true_eigs - current_est);
    err = min(dist);
    err_with_shift = [err_with_shift, err];
    
    if err < 1e-14
        break;
    end
end

%% Plotting
figure('Name', 'Task 2: QR Iteration Convergence - Fixed');
semilogy(1:length(err_no_shift), err_no_shift, 'b-o', 'LineWidth', 1.5, ...
    'MarkerSize', 6, 'DisplayName', 'Without Shift');
hold on;
semilogy(1:length(err_with_shift), err_with_shift, 'r-^', 'LineWidth', 1.5, ...
    'MarkerSize', 6, 'DisplayName', 'With Rayleigh Shift');
xlabel('Iteration', 'FontSize', 12);
ylabel('Error (Distance to closest eigenvalue)', 'FontSize', 12);
title('QR Iteration Convergence Comparison', 'FontSize', 14);
legend('Location', 'best', 'FontSize', 11);
grid on;


fprintf('Iterations without shift: %d\n', length(err_no_shift));
fprintf('Iterations with shift: %d\n', length(err_with_shift));
fprintf('Final error without shift: %.2e\n', err_no_shift(end));
fprintf('Final error with shift: %.2e\n', err_with_shift(end));
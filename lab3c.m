%% Task 3: Algebraic Riccati Equation Solver
    clc; clear; close all;

    % Define Matrices
    A = [0, 1; -2, -3];
    G = [1, 0; 0, 1];
    F = [0, 0; 0, 1];

    % Call the solver function
    [X, eigs_closed_loop, eigs_ham] = solve_are_general(A, G, F);

    % Display Result
    disp('Solution X:');
    disp(X);
    
    % Verify ARE equation: G + XA + A'X - XFX = 0
    are_resid = G + X*A + A'*X - X*F*X;
    disp('ARE Residual (Norm):');
    disp(norm(are_resid));

    % Plotting
    figure();
    % figure 1. Hamiltonian Eigenvalues
    plot(real(eigs_ham), imag(eigs_ham), 'ko', 'MarkerSize', 8, 'LineWidth', 1.5);
    xline(0, '--k'); yline(0, '--k');
    title('Eigenvalues of Hamiltonian Matrix');
    xlabel('Real Part'); ylabel('Imaginary Part');
    grid on; axis equal;

    figure();
    % figure 2.  Closed Loop Eigenvalues (A - FX)
    plot(real(eigs_closed_loop), imag(eigs_closed_loop), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
    xline(0, '--k'); yline(0, '--k');
    title('Eigenvalues of A - FX (Closed Loop)');
    xlabel('Real Part'); ylabel('Imaginary Part');
    grid on; axis equal;
    

function [X, eigs_closed_loop, eigs_ham] = solve_are_general(A, G, F)
    % Inputs: A (nxn), G (symmetric), F (symmetric)
    % Solves G + XA + A'X - XFX = 0
    
    [n, ~] = size(A);
    
    % 1. Construct Hamiltonian Matrix
    % Standard form is [A, -F; -G, -A']. 
    H = [A,   -F; 
         -G,  -A']; 
     
    eigs_ham = eig(H);
    
    % 2. Schur Decomposition
    [U, T] = schur(H);
    
    % 3. Reorder Schur Decomposition
    [U_sorted, T_sorted] = ordschur(U, T, 'lhp');
    
    % 4. Partition U
    U11 = U_sorted(1:n, 1:n);
    U21 = U_sorted(n+1:2*n, 1:n);
    
    % 5. Compute X
    % X = U21 * inv(U11)
    % Ideally solved via linear system / slash operator for stability
    X = U21 / U11;
    X = real((X + X') / 2);
    
    % 6. Compute closed loop eigenvalues
    eigs_closed_loop = eig(A - F*X);
end
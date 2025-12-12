function [ x,myout ] = myADMMp( A,b, myinp)
% solving L1/L2 problem
%           rho * norm(x,1) + 0.5 * sum-square(Ax-b)
% Input:
%   A,b: problem data
% myinp:
%        - myinp.rho 
%        - myinp.tol
%        - myinp.nonorth

%% parameter setting
alpha=0.35;
maxiter = 500;
gamma = 1.199;
rho = myinp.rho;
bmax = norm(b,inf);
tol = myinp.tol/bmax;
nonorth = myinp.nonorth;
beta = 2.5;


if nonorth
    [L,U] = factor(A, beta);
end



%% get problem size
[~, n] = size(A);       nb = norm(b);   sn = sqrt(n);
z = randn(n,1);
y = randn(n,1);

%% scaling problem
b1 = b/bmax;
Atb1 = A'*b1;
x = Atb1;


iter = 0;
for i = 1:maxiter
    iter = iter + 1;
    x_hold = x;
    
    % x-update
    x = Atb1 + beta * z - y;
    if nonorth
        x = x/beta - (A'*(U \ ( L \ (A*x) )))/beta^2;
    else
        x = 1/beta * x - 1/(beta + beta^2) * (A' * (A* x));
    end
    
    % z-update
    hat_x = alpha * x + (1-alpha)*z;
    tmp = hat_x + y/beta;
    z = shrinkage(tmp, rho/beta);
    
    % y-update
    y = y + gamma * beta * (hat_x - z);
    
    %evaluation
    if (norm(x-x_hold)< tol*0.1*norm(x_hold)), break;end
end

%% re-scaling for x
x = x * bmax;
myout.iter = iter;
end


function z = shrinkage(x, kappa)
    z = max(abs(x)-kappa,0) .* sign(x);
end
function [L,U] = factor(A, beta)
    [m, ~] = size(A);
    L = chol( speye(m) + 1/beta*(A*A'), 'lower' );
    L = sparse(L);
    U = sparse(L');
end
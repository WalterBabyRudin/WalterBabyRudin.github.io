function [ x,myout ] = myADMMd( A,b, myinp)
% solving L1/L2 problem
%           rho * norm(x,1) + 0.5 * sum-square(Ax-b)
% Input:
%   A,b: problem data
% myinp:
%        - myinp.rho 
%        - myinp.tol
%        - myinp.nonorth

%% parameter setting
[m, n] = size(A);
rho = myinp.rho;
beta = mean(abs(b));
gamma = 1.618;
tol = myinp.tol;
nonorth = myinp.nonorth;
maxiter = 500;
iter = 1;
x = A'*b;
y = zeros(m,1);
z = zeros(n,1);

Aty = A'*y;
rdbeta = rho/beta;
bdbeta = b/beta;
rdbeta1 = rdbeta + 1;

a = 0;
for i = 1:maxiter
    y_p = y;
    xdbeta = x / beta;
    
    
    if nonorth
        m = 0.5 * (1+sqrt(1+4*a^2)); a_p = a; a = m;
        t = (a_p - 1) / a;
        
        ry = A*(Aty - z + xdbeta) - bdbeta;
        ry = ry + rdbeta*y;
        if iter <= 1
            stp = 1/(3 + rdbeta);
            y = y - stp*ry;
        else
            hat_y = (1+t) * y - t * y_p;
            y_p = y;
            y = hat_y - stp*ry;
        end
    else
        y = A*(z - xdbeta) + bdbeta;
        y = y / rdbeta1;
    end
    Aty = A'*y;
    
    
    z = xdbeta + Aty;
    z = z ./ max(1,abs(z));
    
    
    rd = Aty - z; xp = x;
    x = x + (gamma*beta) * rd;


    if (norm(x-xp)< tol*norm(xp)), break;end
    iter = iter + 1;
end


myout.iter = iter;
end
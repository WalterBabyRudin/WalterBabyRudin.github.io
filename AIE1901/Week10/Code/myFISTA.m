function [ x, myout ] = myFISTA( A,b,myinp )
% [~, n] = size(A);
maxiter = 200;
rho = myinp.rho;
tol = myinp.tol;

x_hold = A'*b;
y = x_hold;
res_y = A*y-b;
grad_y = A'*res_y;

f = @(x,res_x) 0.5*sum_square(res_x) + rho * norm(x,1);
obj_prev = f(y,res_y);

if myinp.nonorth == 1
    alpha = 0.3;
else
    alpha = 1;
end
a = 0;
for iter = 1:maxiter
    
    x = shrinkage(y - alpha*grad_y, alpha*rho);
    m = 0.5 * (1+sqrt(1+4*a^2)); a_p = a; a = m;
    t = (a_p - 1) / a;
    
    y = (1+t) * x - t * x_hold;
    
    x_hold = x;
    
    res_y = A*y-b;
    grad_y = A'*res_y;
    obj = f(y,res_y);
    if (iter > 1) && (abs(obj - obj_prev) <= tol * abs(obj_prev))
        break;
    end
    obj_prev = obj;
end
myout.iter = iter;
x = y;
end

function z = shrinkage(x, kappa)
    z = max(abs(x)-kappa,0) .* sign(x);
end
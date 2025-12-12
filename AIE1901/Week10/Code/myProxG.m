function [ x, myout ] = myProxG( A,b,myinp )

maxiter = 200;
rho = myinp.rho;
tol = myinp.tol;
x = A'*b;
res_x = A*x-b;
grad = A'*res_x;
f = @(x,res_x) 0.5*sum_square(res_x) + rho * norm(x,1);
obj_prev = f(x,res_x);

if myinp.nonorth == 1
    alpha = 0.35;
else
    alpha = 1;
end

a = 0;


for iter = 1:maxiter
    z = shrinkage(x - alpha*grad, alpha*rho);
    x = z;
    res_x = A*x-b;
    obj = f(x,res_x);
    if (abs(obj - obj_prev) <= tol * abs(obj_prev))
        break;
    end
    obj_prev = obj;

    grad = A'*res_x;
end
myout.iter = iter;
end



function z = shrinkage(x, kappa)
    z = max(abs(x)-kappa,0) .* sign(x);
end
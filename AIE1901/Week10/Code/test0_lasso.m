% L1-regularized least-squares example

% test problem sizes
L = 1000;
ns = [1 5:5:15]*L;

orth = input('Test for A*A'' = I (0 or 1)? ');
if isempty(orth), orth = 0; end
opts.nonorth = not(orth);
sigma = 0.05;
rng(0)

for i = 1:numel(ns)
    
    %% Specify problem sizes
    n = ns(i);              % number of features
    m = n/2;                % number of examples
    k = m/5;                % number of nonzeros
    
    %% Generate problem data
    p = randperm(n);
    x0 = zeros(n,1);
    x0(p(1:k)) = randn(k,1);
    
    if opts.nonorth
            A = randn(m,n);
            d = 1./sqrt(sum(A.^2))'; 
            A = A*spdiags(d,0,n,n); % normalize columns
    else
            A=dctmtx(n); A = A(p(1:m),:);
    end
            
    b = A*x0; v = randn(m,1);
    b = b + sigma*norm(b)/norm(v)*v;
    
    rho_max = norm( A'*b, 'inf' );
    rho = 0.1*rho_max;
    opts.rho = rho;
    obj = @(x)sum_square(A*x-b)/2+rho*norm(x,1);
    fprintf('\n')
    
    %% Solve problem    
    opts.tol = 2e-4;
    t0 = tic; [x1,out1] = yall1(A, b, opts);
    t1 = toc(t0); f1 = obj(x1); it1 = out1.iter;

    %% Display stats
    fprintf('\n[n m k] = [%i %i %i]: \n',n,m,k)
    fprintf('-----------------------------------------------------\n');
    fprintf(' YALL1 :  iter = %3i,  f = %f,  time = %6.3f\n',it1,f1,t1)

    if exist('yzProxG','file')
        opts.tol = 1e-6;
        t0 = tic; [x2,out2] = yzProxG(A, b, opts);
        t2 = toc(t0); f2 = obj(x2); it2 = out2.iter;
        fprintf('yzProxG:  iter = %3i,  f = %f,  time = %6.3f\n',it2,f2,t2)
    end
    
    if exist('myADMMd','file')
        opts.tol = 1e-6;
        t0 = tic; [x3,out3] = myADMMd(A, b, rho);
        t3 = toc(t0); f3 = obj(x2); it3 = numel(out3.objval);
        fprintf('myADMMp:  iter = %3i,  f = %f,  time = %6.3f\n',it3,f3,t3)
    end
    
    fprintf('-----------------------------------------------------\n');
    
    %if i == 1, plot(1:n,x1,1:n,x2,'o',1:n,x3,'*'), shg, end
end
fprintf('\n\n')
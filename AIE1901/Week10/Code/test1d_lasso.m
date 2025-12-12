% test script for soling LASSO problem on 1d synthetic data
% ------------------------------------------------------------------
% Codes may use differenbt stopping rules, so one should adjust the 
% tolerance values to achieve a similar accuracy in comparing codes.
% ------------------------------------------------------------------
% (Yin Zhang, s2020)


% clean up
clear, close all, rng(0)

% first run instructor's 3 solvers
Solvers = {' yall1 ','yzFISTA','yzProxG'};
Tols = [5e-3, 1e-5, 2e-5];
%Tols = [1e-3, 1e-3, 1e-3, 1e-5,1e-6];
% --------------- add your 2 solver ----------------------
% add my_solver of ADMM type 
my_solver = 'myADMMd';
if exist(my_solver,'file')
    Solvers = [Solvers,my_solver];
    Tols = [Tols 1e-5];
end
% add my_solver of Proxmal type
my_solver = 'myFISTA';
if exist(my_solver,'file')
    Solvers = [Solvers,my_solver];
    Tols = [Tols 1e-5];
end
% --------------------------------------------------------

% test problem sizes
L = 500*2;
%Ns = (2:2:10)'*L;

Ns = (6:2:10)'*L;


n_Sol = numel(Solvers);
F = zeros(n_Sol,1);
T = F; It = F;
Ts = zeros(numel(Ns),n_Sol);
Str = ['test1d: AA''=/=I';'test1d: AA'' = I'];

for orth = 1
    
    opts.nonorth = not(orth);
    
    for i = 1:numel(Ns)
        
        %% Specify problem sizes
        n = Ns(i);              % number of features
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
            A = dctmtx(n); A = A(p(1:m),:);
        end
        
        b = A*x0; v = randn(m,1); sigma = 0.05;
        b = b + sigma*norm(b)/norm(v)*v;
        
        rho_max = norm( A'*b, 'inf' );
        rho = 0.1*rho_max;
        opts.rho = rho;
        obj = @(x)norm(A*x-b)^2/2+rho*norm(x,1);
        fprintf('\n')
        
        %% Solve problem
        for j = 1:n_Sol
            solver = Solvers{j};
            if ~exist(strtrim(solver),'file'), continue; end
            if j > 1, fprintf([solver ' is solving Lasso\n']), end
            opts.tol = Tols(j);
            t0 = tic; [x,out] = feval(strtrim(solver),A,b,opts);
            T(j) = toc(t0); F(j) = obj(x); It(j) = out.iter;
        end
        
        %% Display stats
        fprintf('\n%s: [n m k] = [%i %i %i]\n',Str(1+orth,:),n,m,k)
        fprintf('-----------------------------------------------------\n');
        for j = 1:n_Sol
            str = [Solvers{j} ':  iter = %3i,  f = %9.5f,  time = %6.3f\n'];
            fprintf(str,It(j),F(j),T(j))
        end
        fprintf('-----------------------------------------------------\n');
        Ts(i,:) = T;
        
    end
    
    fprintf('\n\n')    
    figure(1+orth)
    plot(Ns,Ts,Ns,Ts,'o','linewidth',2);
    xlabel('Size n'); ylabel('Time')
    title(Str(1+orth,:)),  grid on
    legend(Solvers,'location','northwest')
    set(gca,'fontsize',16), shg
end

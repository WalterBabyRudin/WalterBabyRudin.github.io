% Test Dual ADMM for Lasso with the benchmark solver YALL1
% phantom image 2D data (Image Processing Toolbox required)
% (Yin Zhang, s2020)

clear; close all

%% specify solvers and accuracy

Solvers = {' yall1 ','yzFISTA'};
Tols = [1e-3 1e-4];
    
% --------------- add your 2 solver ----------------------
% add my_solver of ADMM type 
my_solver = 'myADMMd';
if exist(my_solver,'file')
    Solvers = [Solvers,my_solver];
    Tols = [Tols 1e-4];
end

% --------------------------------------------------------
ns = numel(Solvers);

%% problem sizes
rng(2019)
imgsize = 128;
xsize = [imgsize imgsize];
n = prod(xsize);
m = floor(0.75*n);

%% exact solution xs
Img = phantom(imgsize);
xs = Img(:);
k = sum(xs ~= 0);

%% set options
opts.rho = 1e-3;
opts.nonorth = 0;
fprintf('\n [n,m,k] = [%i,%i,%i]  rho = %.2e\n',...
    n,m,k,opts.rho);
dashes = '-----------------';
fprintf([dashes dashes dashes '\n']);

%% generate data A and b
p = randperm(n); p = p(1:m); 
if all(p > 1), p(1) = 1; end
A = dctmtx(n)'; A = A(p,:);
sigma = 0.001; 
b = A*xs + sigma*randn(m,1);

for i = 1:ns
    
    % call a solver
    solver = Solvers{i};
    if ~exist(strtrim(solver),'file'), continue, end
    if solver(end) == 'G', opts.tol = opts.tol/10; end
    t0 = tic;
    opts.tol = Tols(i);
    [x,Out] = feval(strtrim(solver), A, b, opts); 
    t1 = toc(t0);
    
    % plot image
    subplot(100+10*ns+i);
    imshow(reshape(x,xsize),[]);
    fprintf([solver ': iter %4i  Rel_err = %6.2e  time %6.2f\n'],...
        Out.iter,norm(x-xs)/norm(xs),t1);
    snrstr = sprintf('snr = %.2f',snr(x,x-xs));
    xlabel(snrstr); title(solver); set(gca,'fontsize',16); shg
    
end
fprintf([dashes '   ' date '   ' dashes '\n\n']);

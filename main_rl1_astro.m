% Test for reweighting l1
% Application to RI imaging 
%
% articles:
% - A. Repetti and Y. Wiaux. A forward-backward algorithm for reweighted procedures: Application to radio-astronomical imaging. In Proceedings of IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Barcelona, Spain, 4-8 Mai 2020
% - A. Repetti, and  Y. Wiaux. Variable Metric Forward-Backward Algorithm for Composite Minimization Problems.  SIAM Journal on Optimization, vol. 31, no. 2, pp. 1215-1241, May 2021.
%
% author: a.repetti@hw.ac.uk









clear all
clc
close all


addpath data
addpath algos
addpath utils

irtdir = '../irt' ; % SET THE SOPT PATH
setup(irtdir);


%%
SNR =@(x,xtrue) 20 * log10(norm(xtrue(:))/norm(xtrue(:)-x(:)));


%% Load image

%load('im_W28_256')
name_im = 'CYN.fits' ;
im_size=[128,256];
[im, N, Ny, Nx] = util_read_image_astro(name_im, im_size);

%% Radio-astro measurements

param_data.input_snr = 20 ;

param_data.p = 0.5 ; 
param_data.sigma = pi/4 ;
param_data.N = N ;

% Generate Gaussian random u-v sampling
[u, v] = util_gen_sampling_pattern('gaussian', param_data);

% Initialize nuFFT operator
% Generate measurment operator with nufft
ox = 2; % oversampling factors for nufft
oy = 2; % oversampling factors for nufft
Kx = 8; % number of neighbours for nufft
Ky = 8; % number of neighbours for nufft
[A, AT, Gw, ~] = op_nufft([v u], [Ny Nx], [Ky Kx], [oy*Ny ox*Nx], [Ny/2 Nx/2]);

param_data.Phi =@(x) Gw * A(x) ;
param_data.Phit =@(y) AT(Gw' * y) ;

% norm of the measurement operator
param_data.normPhi = op_norm(param_data.Phi, param_data.Phit, [Ny, Nx], 1e-4, 200, 0);    


% Generate noisy measurements
y0 = param_data.Phi(im);
sigma_noise = norm(y0(:)) * 10^(-param_data.input_snr/20) / sqrt(numel(y0)) ;


%%

wlt_basis = {'db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'self'};
[param.Wt, param.W] = op_sp_wlt_basis(wlt_basis, 4, Ny, Nx);
param.max_outer = 1500 ;
param.stopit = 2e-4 ; 
param.stopcrit = 1e-3 ;

% ------------------------------------------------------
% ------------------------------------------------------
% CYGA - 128x256 -- some optimised parameters
% ------------------------------------------------------
% % CYGA - 128x256 - iSNR= 30 - p= 0.8 - sigma= pi/4
% lambda_approx=2e-6 ; 
% lambda_exact = 2e-2 ; 
% param.epsilon = 1e-5 ;
% param.lambda_fid = 1e-3 / param_data.normPhi ;
% ------------------------------------------------------
% % CYGA - 128x256 - iSNR= 30 - p= 0.5 - sigma= pi/4
% lambda_approx=3e-6 ; 
% lambda_exact = 2e-2 ; 
% param.epsilon = 1e-5 ;
% param.lambda_fid = 1e-3 / param_data.normPhi ;
% ------------------------------------------------------
% CYGA - 128x256 - iSNR= 20 - p= 0.5 - sigma= pi/4
lambda_approx=3e-5 ; 
lambda_exact = 2e-2 ; 
param.epsilon = 1e-5 ;
param.lambda_fid = 1e-3 / param_data.normPhi ;
% ------------------------------------------------------
% ------------------------------------------------------


param.Adiag = param.lambda_fid* param_data.normPhi ;
param.gamma = 0.9 ;
param.display = 50 ;
param.xtrue = im ; % for SNR computation
Phi = param_data.Phi ;
Phit = param_data.Phit ;
isnr = param_data.input_snr ;

Seeds = 5 ;
Max_inner = [2, 5,10, 15, 20, 30, 50, 70] ; 
Res = cell(numel(Max_inner),Seeds) ;
Res_comp = cell(Seeds) ;

for seed=1:Seeds
    rng(seed) ;
    
    disp(' ')
    disp('******************************************')
    disp(['Test seed = ',num2str(seed)])
    disp('******************************************')
    
    

    noise = (randn(size(y0)) + 1i*randn(size(y0))) * sigma_noise /sqrt(2) ;
    y = y0 + noise ;
    x0 = zeros(size(Phit(y))) ;
    
    disp(['SNR observation: ',num2str(SNR(y,y0))])
    disp(['SNR initialization: ',num2str(SNR(x0,im))])
    disp(' ')

%%    
    % Reweighting tests ---------------------------------------------------
    param.lambda = lambda_approx ;
    for mm = 1:numel(Max_inner)
        param.max_inner = Max_inner(mm) ;
        Res{mm,seed} = ApproxFB_RWL1_pos(x0, y, Phi, Phit, param) ;
        snr_save(mm,seed) = Res{mm,seed}.SNR(end) ;
        nb_it(mm,seed) = numel(Res{mm,seed}.SNR) ;
        crit_save(mm,seed) = Res{mm,seed}.crit(end) ;
        time_save(mm,seed) = sum(Res{mm,seed}.time_tot) ;
    end
%%    
    
end



%%

display_fig = 1 ;

if display_fig

%%
figure,
plot(Max_inner, mean(snr_save,2),'o:'), 
xlabel('nb inner it'), ylabel('SNR'), 

figure
plot(Max_inner, mean(nb_it,2),'o:'), 
xlabel('nb inner it'), ylabel('total nb it'), 

figure
plot(Max_inner, mean(nb_it,2),'o:'), 
xlabel('nb inner it'), ylabel('Time reconstruction'), 

end




%%


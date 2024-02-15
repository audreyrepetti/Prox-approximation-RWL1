function results = ApproxFB_RWL1_pos(x0, y, Phi, Phit, param)

% param: W, Wt, lambda, Adiag, epsilon, gamma

paramL1.Psit = param.W ;
paramL1.Psi = param.Wt ;
paramL1.tight = 0 ;
paramL1.verbose = 0 ;
paramL1.pos = 1 ;
paramL1.rel_obj = 1e-1 ;

x = x0 ;

SNR =@(x,xtrue) 20 * log10(norm(xtrue(:))/norm(xtrue(:)-x(:)));
SNRit(1) = SNR(x,param.xtrue) ;
SNRlog(1) = SNR(log10(x+1e-5),log10(param.xtrue+1e-5)) ;
res = Phi(x) - y ;
Wx = param.W(x) ;
reg(1) = param.lambda * sum( log( abs(Wx)+param.epsilon ) ) ;
fid(1) = 0.5 * param.lambda_fid * sum(abs(res(:)).^2) ;
crit(1) = fid(1) + reg(1) ;
paramL1.weights = param.lambda ;
res_it(1) = 0 ;
res_crit(1) = 0 ;

for iter = 1:param.max_outer
    xold = x ;
    
    tic ;
    if mod(iter, param.max_inner)==0
        Wx = param.W(x) ;
        paramL1.weights = param.lambda./ (abs(Wx) + param.epsilon) ;
    end
    
    grad = real(Phit(res)) ;
    xgrad = x - param.gamma * param.Adiag.^(-1).*param.lambda_fid .* grad ;
    x = solver_prox_L1(xgrad, param.gamma^(-1) * param.Adiag(:), paramL1) ;
    time_tot(iter) = toc ;
    
    res = Phi(x)-y ;
    fid(iter+1) = 0.5 * param.lambda_fid * sum(abs(res(:)).^2) ;
    Wx = param.W(x) ;
    reg(iter+1) = param.lambda * sum( log( abs(Wx)+param.epsilon ) ) ;
    crit(iter+1) = fid(iter+1) + reg(iter+1) ;
    SNRit(iter+1) = SNR(x,param.xtrue) ;
    SNRlog(iter+1) = SNR(log10(x+1e-5),log10(param.xtrue+1e-5)) ;
    
    if mod(iter,param.display) == 0
        figure(100)
        subplot 231
        imagesc(log10(x)), axis image, colormap jet, colorbar, caxis([-3.5,max(log10(x(:)))])
        xlabel(['it = ', num2str(iter),' -- SNR = ', num2str(SNRit(iter+1))])
        subplot 232
        semilogy(res_it), 
        xlabel('iterations'), ylabel('residual iterate norms'), %axis([0, param.max_outer, log(min(res_it)), log(max(res_it))])
        subplot 233
        semilogy(res_crit), 
        xlabel('iterations'), ylabel('residual obj. value'), 
        subplot 234
        plot(SNRit), 
        xlabel('iterations'), ylabel('SNR'), 
        subplot 235
        plot(fid), 
        xlabel('iterations'), ylabel('data term'), 
        subplot 236
        plot(reg), 
        xlabel('iterations'), ylabel('reg. term'), 
        
        pause(0.1)
    end
    
    res_it(iter+1) = norm(x(:)-xold(:))/norm(x) ;
    res_crit(iter+1) = abs(crit(iter+1)-crit(iter))/abs(crit(iter+1)) ;
    if iter>param.max_inner && res_it(iter+1)<param.stopit ...
            && res_crit(iter+1)<param.stopcrit
        break
    end
%     if res_it(iter+1)< 1e-6 ...
%             || res_crit(iter+1)< 1e-6
%         break
%     end
end

disp('----------------------------')
disp(['nb inner it  : ',num2str(param.max_inner)])
disp(['nb total it  : ',num2str(iter)])
disp(['residual it  : ',num2str(res_it(end))])
disp(['residual crit: ',num2str(res_crit(end))])
disp(['SNR          : ',num2str(SNRit(iter+1))])
disp(['SNR log im    : ',num2str(SNRlog(iter+1))])
disp(['crit          : ',num2str(crit(iter+1))])
disp(['Comp. time    : ',num2str(sum(time_tot))])
disp('----------------------------')

results.x = x ;
results.SNR = SNRit ;
results.SNRlog = SNRlog ;
results.fid = fid ;
results.reg = reg ;
results.crit = crit ;
results.res_crit = res_crit ;
results.res_it = res_it ;
results.time_tot = time_tot ;


end
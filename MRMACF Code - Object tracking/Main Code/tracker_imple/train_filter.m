function [filter_model_f, spatial_selection, channel_selection] = train_filter(xlf, feature_info, yf, seq, params, filter_model_f,reg_window,xlf_u,xlf_l,xlf_d,xlf_r)

if seq.frame == 1
    filter_model_f = cell(size(xlf));
    channel_selection = cell(size(xlf));
    spatial_selection = params.mask_window;
    xi_f=cell(size(xlf));
  
end
r_f=cell(size(xlf));


for k = 1 : numel(xlf) 
    model_xf = xlf{k};
    if(seq.frame == 1)               
        extend_xf=cat(4,model_xf, params.back_imp*xlf_u{k} , params.back_imp*xlf_l{k} , params.back_imp*xlf_d{k} , params.back_imp*xlf_r{k} );
        extend_xf=extend_xf(:,:,:,1);
    else
        model_xf = ((1 - params.learning_rate(feature_info.feature_is_deep(k)+1)) * model_xf) + (params.learning_rate(feature_info.feature_is_deep(k)+1) * xlf{k});
        extend_xf=cat(4,model_xf, params.back_imp*xlf_u{k} , params.back_imp*xlf_l{k} , params.back_imp*xlf_d{k} , params.back_imp*xlf_r{k} );
        extend_xf=extend_xf(:,:,:,1);
    end  
     
    r_f{k} = yf{k};  
    rp_f{k} = yf{k};  
    % intialize the variables and parameters
    if (seq.frame == 1) %
        filter_model_f{k} = zeros(size(model_xf));
        lambda3 = 1e-3;
        iter_max = 3;
        mu_max = 20;                
    else
%         feature_info.feature_is_deep(k)+1
                
        lambda3 = params.lambda3(feature_info.feature_is_deep(k)+1);
%         disp(lambda3)
        iter_max = 2;
        mu_max = 0.1;        
    end

    xi_f{k} = single(zeros(size(yf{k})));
    update_interval=1;

    filter_f = single(zeros(size(model_xf)));
    filter_prime_f = filter_f;
    gamma_f = filter_f;
    mu  = 1; 
    
    % pre-compute the variables
    
    T = feature_info.data_sz(k)^2;
%     S_xx = sum(conj(model_xf) .* model_xf, 3);    
    Sfilter_pre_f = sum(conj(model_xf) .* filter_model_f{k}, 3);
   
    Sfx_pre_f = bsxfun(@times, model_xf, Sfilter_pre_f);
    
    iter = 1;
    
    d=ones(feature_info.data_sz(k),1);
    
    if (seq.frame==1||mod(seq.frame,update_interval)==0)  
   
    while (iter <= iter_max)       
       
        S_xx = sum(conj(extend_xf) .* extend_xf, 3);
        
        D = S_xx + T * (mu + lambda3);
        
        Spx_f = sum(conj(extend_xf) .* filter_prime_f, 3);
        Sgx_f = sum(conj(extend_xf) .* gamma_f, 3);
        
        filter_f = ((1/(T*(mu + lambda3)) * bsxfun(@times,  params.omega_f{k}, model_xf)) - ((1/(mu + lambda3)) * gamma_f) +(mu/(mu + lambda3)) * filter_prime_f) + ...
        (lambda3/(mu + lambda3)) * filter_model_f{k} - bsxfun(@rdivide,(1/(T*(mu + lambda3)) * bsxfun(@times, model_xf, (S_xx .*  params.omega_f{k})) + ...
        (lambda3/(mu + lambda3)) * Sfx_pre_f - (1/(mu + lambda3))* (bsxfun(@times, extend_xf, Sgx_f)) +(mu/(mu + lambda3))* (bsxfun(@times, extend_xf, Spx_f))), D);
                                                                            

        if iter == iter_max && seq.frame > 1
            break;
        end
 
        pmu = ifft2((mu * filter_f+ gamma_f), 'symmetric');
             
        % Spatial-Regularization Term
        spatial_term =reg_window{k}./pmu;
           if (seq.frame == 1)
            filter_prime = zeros(size(pmu));
            channel_selection{k} = ones(size(pmu));
            for i = 1:size(pmu,3)
                filter_prime(:,:,i) = spatial_selection{k} .* pmu(:,:,i);
            end
        else
            filter_prime = pmu;
            if feature_info.feature_is_deep(k)
                channel_selection{k}=[];
                spatial_selection_3;
                spatial_selection{k}=[];   
            else                
                spatial_selection_2;           
                spatial_selection{k}=[];
                channel_selection{k}=[];               
            end
        end
        filter_prime_f = fft2(filter_prime);      
        gamma_f = gamma_f + (mu * (filter_f - filter_prime_f));  
        mu = min(1.5 * mu, mu_max);
        iter = iter+1;
    end

    iter1 = 1;    
    params.admm_iterations = 2;
    gamma = 27;
    lambda_2 = 840;
    eta   = 0.044;
    betha = 10;
    phi=1;
    gammamax = 10000;

     while(iter1 <= iter_max)
     %solve for s
         S_gx = sum(bsxfun(@times, conj(pmu), model_xf), 3);                              
         s_f  = 1/(1+gamma*T)*(S_gx + gamma*T*r_f{k} + T*xi_f{k});                    
     %solve for r
         %r_f   = 1/(lambda_2*(1+Psi^2)+phi*(1-Psi^2)+gamma*T)*(lambda_2*(1+Psi^2)*(yf + eta*(1-Psi)* y_2_f) + phi*(1-Psi^2)*rp_f + gamma*T*s_f - T*xi_f);  
         r_f{k}   = 1/(lambda3+phi+gamma*T)*(lambda3*params.omega_f{k} + phi*rp_f{k} + gamma*T*s_f - T*xi_f{k});            
%          r_f{k}   = 1/(lambda3+gamma*T)*(lambda3*(yf{k} + eta*params.y_2_f{k}) + gamma*T*s_f - T*xi_f{k});         
     %update Xi
         xi_f{k}  = xi_f{k} + gamma*(s_f - r_f{k});                                
     %    update gamma- betha =10.
         gamma = min(betha * gamma, gammamax);
         iter1    = iter1+1; 
     end   
    
    end    
    % update the filters Eqn.2
    % <
    if seq.frame == 1
        filter_model_f{k} = filter_f;     
    else
        filter_model_f{k} = params.learning_rate(feature_info.feature_is_deep(k)+1)*filter_f ...
            + (1-params.learning_rate(feature_info.feature_is_deep(k)+1))*filter_model_f{k};     
    end
    % >
end

end


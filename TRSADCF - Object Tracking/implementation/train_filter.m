function [filter_model_f, filter_reg_g] = train_filter(xlf, feature_info, yf, seq, params,num_feature_blocks,filter_model_f,my_response_pwp, filter_reg_g)

if seq.frame == 1
    filter_model_f = cell(size(xlf));
    filter_reg_g = cell(size(xlf));
    channel_selection = cell(size(xlf));
end

for k = 1: numel(xlf)
    model_xf = gather(xlf{k});    
    if (seq.frame == 1)
        filter_reg_g{k} = zeros(size(model_xf));
        filter_model_f{k} = zeros(size(model_xf));
        lambda2 = 0;        
    else
        lambda2 = params.temporal_consistency_factor(feature_info.feature_is_deep(k)+1);        
    end
    mu_max = 0.1;
    % intialize the variables
    g_g = single(zeros(size(model_xf)));
    f_prime_f = g_g;
    Gamma_f = g_g;
    mu  = params.init_penalty_factor;
    mu_scale_step = params.penalty_scale_step;
    
    s_x=sum(conj(model_xf) .* model_xf,3);            
    s_y=sum(conj(model_xf) .* yf{k},3);
    omega = cell(numel(num_feature_blocks), 1);   
      neeta=0.0000000003; 
    
    % pre-compute the variables
    T = feature_info.data_sz(k)^2;
    S_xx = sum(conj(model_xf) .* model_xf, 3);
    
    Stheta_pre_f = sum(conj(model_xf) .* filter_reg_g{k}, 3);
    Sfx_pre_f = bsxfun(@times, model_xf, Stheta_pre_f);
    
    % solve via ADMM algorithm
    iter = 1;
    while (iter <= params.max_iterations)           
        B = S_xx + T * (mu + lambda2);
        Sgx_f = sum(conj(model_xf) .* f_prime_f, 3);
        Shx_f = sum(conj(model_xf) .* Gamma_f, 3);
         
        %sub-problem-g:
        g_g = ((1/(T*(mu + lambda2)) * bsxfun(@times,  yf{k}, model_xf)) - ((1/(mu + lambda2)) * Gamma_f) +(mu/(mu + lambda2)) * f_prime_f)+ (lambda2/(mu + lambda2)) * filter_reg_g{k}- ...
            bsxfun(@rdivide,(1/(T*(mu + lambda2)) * bsxfun(@times, model_xf, (S_xx .*  yf{k}))+ (lambda2/(mu + lambda2)) * Sfx_pre_f - ...
            (1/(mu + lambda2))* (bsxfun(@times, model_xf, Shx_f)) +(mu/(mu + lambda2))* (bsxfun(@times, model_xf, Sgx_f))), B);


        %sub-problem-h: 
        if (seq.frame == 1)          
          h_h = ((T/((mu*T)))+ (params.admm_lambda)+(neeta*s_x)) .* ifft2((mu*g_g) + Gamma_f + (neeta*s_y)); 
            h = real(h_h);            
            T1 = zeros(size(h));
            channel_selection{k} = ones(size(h));
            for i = 1:size(h,3)
                T1(:,:,i) = params.mask_window{k} .* h(:,:,i);
            end
        else
            omega{k}=sum(conj(my_response_pwp{k}) .* my_response_pwp{k},3);
            h_h = ((T/((mu*T)))+ (params.admm_lambda*omega{k})+(neeta*s_x)) .* ifft2((mu*g_g) + Gamma_f + (neeta*s_y));                                 
            h = real(h_h);
            T1=h;            
            channel_selection{k} = max(0,1-5./(numel(h)*sqrt(sum(sum(T1.^2,1),2))));
            
            [~,b] = sort(channel_selection{k}(:),'descend');
            
            if numel(b)>1
                channel_selection{k}(b(ceil(feature_info.channel_selection_rate(k)*numel(b)):end)) = 0;
            end            
            T1 = repmat(channel_selection{k},size(T1,1),size(T1,2),1) .* T1;
        end
                   
        f_prime_f = fft2(T1);
                    
        
        %   update Gamma
        Gamma_f = Gamma_f + (mu * (g_g - f_prime_f));
        
        %   update mu
        mu = min(mu_scale_step * mu,mu_max);
        
        iter = iter+1;
    end
    
    % save the trained filters
    filter_reg_g{k} = g_g+randn(size(g_g))*mean(g_g(:))*params.stability_factor;
    
    if seq.frame == 1
        filter_model_f{k} = g_g;
    else
        filter_model_f{k} = feature_info.learning_rate(k)*g_g + (1-feature_info.learning_rate(k))*filter_model_f{k};
    end
end
end
%   





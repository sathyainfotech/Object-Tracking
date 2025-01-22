function [results,output] = tracker_main(params)
% Run tracker
% Input  : params[structure](parameters and sequence information)
% Output : results[structure](tracking results)

% Get sequence information
% <
[seq, im] = get_sequence_info(params.seq);
params = rmfield(params, 'seq');

if isempty(im)
    seq.rect_position = [];
    [~, results] = get_sequence_results(seq);
    return;
end

seq = get_sequence_vot(seq);

pos = seq.init_pos(:)';
target_sz = seq.init_sz(:)';
ratio_sz = target_sz(1)/target_sz(2);
params.init_sz = target_sz;
features = params.t_features;
params = init_default_params(params);
init_target_sz = target_sz;

%-SATHYA
sz_ratio = params.sz_ratio; 
delta    = params.delta;
nu       = params.nu;
eta      = params.eta;
half_sub_cen    = 6;
cen = 2;

offset_prop=params.offset_prop;
back_imp=params.back_imp;
% >

% Set Global parameters and data type
% <
if isfield(params, 't_global')
    global_fparams = params.t_global;
else
    global_fparams = [];
end
global_fparams.use_gpu = params.use_gpu;
global_fparams.gpu_id = params.gpu_id;
global_fparams.augment = 0;

if params.use_gpu
    if isempty(params.gpu_id)
        gD = gpuDevice();
    elseif params.gpu_id > 0
        gD = gpuDevice(params.gpu_id);
    end
    params.data_type = zeros(1, 'single', 'gpuArray');
else
    params.data_type = zeros(1, 'single');
end
params.data_type_complex = complex(params.data_type);

global_fparams.data_type = params.data_type;
% >

% Check if color image
% <
if size(im,3) == 3
    if all(all(im(:,:,1) == im(:,:,2)))
        is_color_image = false;
    else
        is_color_image = true;
    end
else
    is_color_image = false;
end

if size(im,3) > 1 && is_color_image == false
    im = im(:,:,1);
end
% >

% Calculate search area and initial scale factor
% <
search_area = prod(params.search_area_scale'*init_target_sz,2);
currentScaleFactor = zeros(numel(search_area),1);
for i = 1 : numel(currentScaleFactor)
    if search_area(i) > params.max_image_sample_size(i)
        currentScaleFactor(i) = sqrt(search_area(i) / params.max_image_sample_size(i));
    elseif search_area(i) < params.min_image_sample_size(i)
        currentScaleFactor(i) = sqrt(search_area(i) / params.min_image_sample_size(i));
    else
        currentScaleFactor(i) = 1.0;
    end
end
% >

% target size at the initial scale
% <
base_target_sz = 1 ./ currentScaleFactor*target_sz;
% >


% search window size
% <
img_sample_sz = repmat(sqrt(prod(base_target_sz,2) .* (params.search_area_scale.^2')),1,2); % square area, ignores the target aspect ratio
% >

% initialise feature settings
% <
[features, global_fparams, feature_info] = init_features(features, global_fparams, params, is_color_image, img_sample_sz, 'odd_cells');
img_support_sz = feature_info.img_support_sz;
feature_sz = feature_info.data_sz;
num_feature_blocks = size(feature_sz, 1);
% >

% Size of the extracted feature maps (filters)
% <
feature_sz_cell = permute(mat2cell(feature_sz, ones(1,num_feature_blocks), 2), [2 3 1]);
filter_sz = feature_sz + mod(feature_sz+1, 2);
filter_sz_cell = permute(mat2cell(filter_sz, ones(1,num_feature_blocks), 2), [2 3 1]);

[output_sz, k1] = max(filter_sz, [], 1);
params.output_sz = output_sz;
k1 = k1(1);
block_inds = 1:num_feature_blocks;
block_inds(k1) = [];
% >

% Pre-computes the grid that is used for socre optimization
% <
ky = circshift(-floor((filter_sz_cell{1}(1) - 1)/2) : ceil((filter_sz_cell{1}(1) - 1)/2), [1, -floor((filter_sz_cell{1}(1) - 1)/2)]);
kx = circshift(-floor((filter_sz_cell{1}(2) - 1)/2) : ceil((filter_sz_cell{1}(2) - 1)/2), [1, -floor((filter_sz_cell{1}(2) - 1)/2)])';
newton_iterations = params.newton_iterations;
% >

% Construct the Gaussian label function, cosine window and initial mask
% <
% offset = [-target_sz(1) -target_sz(2); -target_sz(1) 0; -target_sz(1) target_sz(2); 
%          0 -target_sz(2); 0 target_sz(2); target_sz(1) -target_sz(2); target_sz(1) 0; target_sz(1) target_sz(2)]; % sub region position define

central_target = cell(num_feature_blocks, 1);
for i = 1 : numel(currentScaleFactor)    
    central_target{i} = base_target_sz/sz_ratio;    
end



yf = cell(num_feature_blocks, 1);
for i = 1:num_feature_blocks
    sz = filter_sz_cell{i};
    output_sigma_factor = params.output_sigma_factor(feature_info.feature_is_deep(i)+1);
    output_sigma  = sqrt(prod(floor(base_target_sz(feature_info.feature_is_deep(i)+1,:))))*feature_sz_cell{i}./img_support_sz{i}* output_sigma_factor;
    rg            = circshift(-floor((sz(1)-1)/2):ceil((sz(1)-1)/2), [0 -floor((sz(1)-1)/2)]);
    cg            = circshift(-floor((sz(2)-1)/2):ceil((sz(2)-1)/2), [0 -floor((sz(2)-1)/2)]);
    [rs, cs]      = ndgrid(rg,cg);
    y             = exp(-0.5 * (((rs.^2 + cs.^2) / mean(output_sigma)^2)));
    yf{i}         = fft2(y);
end

% Construct 1st direction square label
    y2 = cell(num_feature_blocks, 1);  
    for i = 1:num_feature_blocks   
        sz = filter_sz_cell{i};         
        central_target{1}(1,1) = max(central_target{1}(1,1),5);
        central_target{2}(1,2) = max(central_target{2}(1,2),5);
        rg2 = ones(1,sz(1));
        cg2 = ones(1,sz(2));    
        [rs2,cs2] = ndgrid(rg2,cg2);
        rs2(:,ceil((central_target{1}(1,1)-1)/2):sz(1)-floor((central_target{1}(1,1)-1)/2)) = 0;
        rs2(ceil((central_target{2}(1,2)-1)/2):sz(1)-floor((central_target{2}(1,2)-1)/2),:) = 0;
        cs2(:,ceil((central_target{1}(1,1)-1)/2):sz(1)-floor((central_target{1}(1,1)-1)/2)) = 0;
        cs2(ceil((central_target{2}(1,2)-1)/2):sz(1)-floor((central_target{2}(1,2)-1)/2),:) = 0;
        y2{i} = ((rs2+cs2)/2);              
    end

%Constrcut 2nd direction square label
    y3 = cell(num_feature_blocks, 1);
    for i = 1:num_feature_blocks
        sz = filter_sz_cell{i};  
        rg3 = ones(1,sz(1));
        cg3 = ones(1,sz(2));    
        [cs3,rs3] = ndgrid(rg3,cg3);
        rs3(:,ceil((central_target{2}(1,2)-1)/2):sz(2)-floor((central_target{2}(1,2)-1)/2)) = 0;
        rs3(ceil((central_target{1}(1,1)-1)/2):sz(1)-floor((central_target{1}(1,1)-1)/2),:) = 0;
        cs3(:,ceil((central_target{2}(1,2)-1)/2):sz(2)-floor((central_target{2}(1,2)-1)/2)) = 0;
        cs3(ceil((central_target{1}(1,1)-1)/2):sz(1)-floor((central_target{1}(1,1)-1)/2),:) = 0;
        y3{i} = ((rs3+cs3)/2);
    end
    
%Constrcut the overlap label
    y4 = cell(num_feature_blocks, 1);
    for i = 1:num_feature_blocks
        sz = filter_sz_cell{i};  
        rg4 = ones(1,sz(1));
        cg4 = ones(1,sz(2));    
        [cs4,rs4] = ndgrid(rg4,cg4);
        min_sz = min(central_target{1}(1,1),central_target{2}(1,2));
        rs4(:,ceil((min_sz-1)/2):sz(2)-floor((min_sz-1)/2)) = 0;
        rs4(ceil((min_sz-1)/2):sz(1)-floor((min_sz-1)/2),:) = 0;
        cs4(:,ceil((min_sz-1)/2):sz(2)-floor((min_sz-1)/2)) = 0;
        cs4(ceil((min_sz-1)/2):sz(1)-floor((min_sz-1)/2),:) = 0;    
        y4{i} = ((rs4+cs4)/2);
    end 

    % Constrcut the hybrid label
    y_2 = cell(num_feature_blocks, 1);
    y_2_f = cell(num_feature_blocks, 1);
    for i = 1:num_feature_blocks
        y_2{i} = ((y2{i}./y2{i}(1,1))+(y3{i}./y3{i}(1,1))-(y4{i}./(y4{i}(1,1))));
        y_2_f{i} = fft2(y_2{i});        
    end
    params.y_2_f=y_2_f;

    % Construct the distance matrix
    sub_cen = 2*half_sub_cen + 1;
    dis = zeros(sub_cen);
    for i=1:sub_cen
        for j=1:sub_cen
            dist_2   = sqrt((i-(half_sub_cen+1))^2 + (j-(half_sub_cen+1))^2);
            dis(i,j) = nu/(1+delta*exp(dist_2));
        end
    end

global_fparams.data_type = params.data_type;

cos_window = cellfun(@(sz) hann(sz(1)+2)*hann(sz(2)+2)', feature_sz_cell, 'uniformoutput', false);
cos_window = cellfun(@(cos_window) cast(cos_window(2:end-1,2:end-1), 'like', params.data_type), cos_window, 'uniformoutput', false);

mask_window = cell(1,1,num_feature_blocks);
mask_search_window = cellfun(@(sz,feature) ones(round(currentScaleFactor(feature.fparams.feature_is_deep+1)*sz)) * 1e-3, img_support_sz', features, 'uniformoutput', false);
target_mask = 1.2*seq.target_mask;
target_mask_range = zeros(2, 2);
for j = 1:2
    target_mask_range(j,:) = [0, size(target_mask,j) - 1] - floor(size(target_mask,j) / 2);
end
for i = 1:num_feature_blocks
    mask_center = floor((size(mask_search_window{i}) + 1)/ 2) + mod(size(mask_search_window{i}) + 1,2);
    target_h = (mask_center(1)+ target_mask_range(1,1)) : (mask_center(1) + target_mask_range(1,2));
    target_w = (mask_center(2)+ target_mask_range(2,1)) : (mask_center(2) + target_mask_range(2,2));
    mask_search_window{i}(target_h, target_w) = target_mask;
    mask_window{i} = mexResize(mask_search_window{i}, filter_sz_cell{i}, 'auto');
end
params.mask_window = mask_window;
% >

% Use the pyramid filters to estimate the scale
% <
nScales = params.number_of_scales;
scale_step = params.scale_step;
scale_exp = (-floor((nScales-1)/2):ceil((nScales-1)/2));
scaleFactors = scale_step .^ scale_exp;
% >

seq.time = 0;
scores_fs_feat = cell(1,1,num_feature_blocks);
response_feat = cell(1,1,num_feature_blocks);
store_filter = [];
while true
    % Read image and timing
    % <
    if seq.frame > 0
        [seq, im] = get_sequence_frame(seq);
        if isempty(im)
            break;
        end
        if size(im,3) > 1 && is_color_image == false
            im = im(:,:,1);
        end
    else
        seq.frame = 1;
        filter_model_f = [];
    end
    
    tic();
    % >
    
    %% Target localization step
    if seq.frame > 1
        global_fparams.augment = 0;
        old_pos = inf(size(pos));
        iter = 1;
        
        %translation search
        while iter <= params.refinement_iterations && any(old_pos ~= pos)
            % Extract features at multiple resolutions
            % <
            sample_pos = round(pos);
            sample_scale = currentScaleFactor*scaleFactors;
            [xt, img_samples] = extract_features(im, sample_pos, sample_scale, features, global_fparams, feature_info);
            
            xtw = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xt, cos_window, 'uniformoutput', false);
            xtf = cellfun(@fft2, xtw, 'uniformoutput', false);
            % >
            
            % Calculate and fuse responses
            % <
            response_handcrafted = 0;
            response_deep = 0;
            for k = [k1 block_inds]
                if feature_info.feature_is_deep(k) == 0
                    scores_fs_feat{k} = gather(sum(bsxfun(@times, conj(filter_model_f{k}), xtf{k}), 3));
                    scores_fs_feat{k} = resizeDFT2(scores_fs_feat{k}, output_sz);
                    response_feat{k} = ifft2(scores_fs_feat{k}, 'symmetric');
                    response_handcrafted = response_handcrafted + response_feat{k};
                else
                    output_sz_deep = round(output_sz/img_support_sz{k1}*img_support_sz{k});
                    output_sz_deep = output_sz_deep + 1 + mod(output_sz_deep,2);
                    scores_fs_feat{k} = gather(sum(bsxfun(@times, conj(filter_model_f{k}), xtf{k}), 3));
                    scores_fs_feat{k} = resizeDFT2(scores_fs_feat{k}, output_sz_deep);
                    response_feat{k} = ifft2(scores_fs_feat{k}, 'symmetric');
                    response_feat{k}(ceil(output_sz(1)/2)+1:output_sz_deep(1)-floor(output_sz(1)/2),:,:,:)=[];
                    response_feat{k}(:,ceil(output_sz(2)/2)+1:output_sz_deep(2)-floor(output_sz(2)/2),:,:)=[];
                    response_deep = response_deep + response_feat{k};
                end
            end
            [disp_row, disp_col, sind, ~, response, w] = resp_newton(squeeze(response_handcrafted)/feature_info.feature_hc_num, squeeze(response_deep)/feature_info.feature_deep_num,...
                newton_iterations, ky, kx, output_sz);
            % >
            
            % Compute the translation vector
            % <
            translation_vec = [disp_row, disp_col] .* (img_support_sz{k1}./output_sz) * currentScaleFactor(1) * scaleFactors(sind);
            if seq.frame < 10
                scale_change_factor = scaleFactors(ceil(params.number_of_scales/2));
            else
                scale_change_factor = scaleFactors(sind);
            end
            % >
            
            % update position
            % <
            old_pos = pos;
            if sum(isnan(translation_vec))
                pos = sample_pos;
            else
                pos = sample_pos + translation_vec;
            end
            
            if params.clamp_position
                pos = max([1 1], min([size(im,1) size(im,2)], pos));
            end
            % >
            
            % Update the scale
            % <
            currentScaleFactor = currentScaleFactor * scale_change_factor;

            %              % Evaluate the effect of side peak 
%                 % current frame
                map_shift = fftshift(response);
                map_sz = size(map_shift);
                [r_max,cen_pos] = max(map_shift(:));
                [rpos,cpos] = ind2sub(size(map_shift),cen_pos);
                 
                if rpos+half_sub_cen+1>map_sz(1)
                    rpos=map_sz(1)-half_sub_cen-1;
                elseif rpos-half_sub_cen-1<0
                    rpos=half_sub_cen+1;
                end
                if cpos+half_sub_cen+1>map_sz(2)
                    cpos=map_sz(2)-half_sub_cen-1;
                elseif cpos-half_sub_cen-1<0
                    cpos=half_sub_cen+1;
                end
  
                sub_cen = map_shift(rpos-half_sub_cen:rpos+half_sub_cen,cpos-half_sub_cen:cpos+half_sub_cen); 
                sub_cen(half_sub_cen : half_sub_cen+cen,half_sub_cen : half_sub_cen+cen) = 0;
                find_local_max = imregionalmax(sub_cen);
                
                local_max = sub_cen.*find_local_max.*dis;
                
                [rsd_max,~] = max(local_max(:));   
                tau = rsd_max/r_max;  

            % >
            iter = iter + 1;
        end
    end

    omega_f =  cell(num_feature_blocks, 1);
    for i = 1:num_feature_blocks         
        if seq.frame == 1
            omega_f{i} = yf{i} ;
           
        else                         
            omega_f{i} = yf{i} + eta*(1-tau)* y_2_f{i};            
        end
    end
    params.omega_f=omega_f;
    
    reg_window = cell(num_feature_blocks, 1);
    for i = 1:num_feature_blocks
        reg_scale = floor(base_target_sz/params.feature_downsample_ratio);
        use_sz = filter_sz_cell{i};    
        wrg = round( ( 1:use_sz(1) ) - use_sz(1)/2 );
        wcg = round( ( 1:use_sz(2) ) - use_sz(2)/2 );
        [wrs,wcs] = ndgrid(wrg , wcg);
        reg_window_temp = params.eta * ( (wrs/base_target_sz(1)) .^ 2 + (wcs/base_target_sz(2)) .^ 2 ) ; %elipse shaped regularization
        critical_value = floor( use_sz/2 + params.regularizer_size * reg_scale/2 );
        reg_window_temp = reg_window_temp - reg_window_temp(critical_value(1) , critical_value(2)) ;
        reg_window_temp(reg_window_temp<0) = 0;
        reg_window_temp = reg_window_temp + params.reg_window_min;
        reg_window_temp(reg_window_temp > params.reg_window_max) = params.reg_window_max;
        reg_window{i} = reg_window_temp;
    end

    %% Model update step
	
    % Extract features and learn filters
    
    global_fparams.augment = 1;

    % Surrounding-Aware
    % extract image region for training sample    
    offset_len=sqrt(prod(target_sz));
    if target_sz(2)>target_sz(1)
        asp_ratio=target_sz(2)/target_sz(1);
        offset_xf=offset_prop;
        offset_yf=offset_xf*sqrt(asp_ratio);
    else
        asp_ratio=target_sz(1)/target_sz(2);
        offset_yf=offset_prop;
        offset_xf=offset_yf*sqrt(asp_ratio);
    end

    offset = floor( [-offset_xf*offset_len 0 ;  0 -offset_yf*offset_len ;  offset_xf*offset_len 0 ;  0 offset_yf*offset_len] );
    sample_pos = round(pos);
    xl = extract_features(im, sample_pos, currentScaleFactor, features, global_fparams, feature_info);
    xl_u = extract_features(im, sample_pos+offset(1,:), currentScaleFactor, features, global_fparams, feature_info);
    xl_l = extract_features(im, sample_pos+offset(2,:), currentScaleFactor, features, global_fparams, feature_info);
    xl_d = extract_features(im, sample_pos+offset(3,:), currentScaleFactor, features, global_fparams, feature_info);
    xl_r = extract_features(im, sample_pos+offset(4,:), currentScaleFactor, features, global_fparams, feature_info);
    
    % do windowing of features
    xlw = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xl, cos_window, 'uniformoutput', false);
    xlw_u = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xl_u, cos_window, 'uniformoutput', false);
    xlw_l = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xl_l, cos_window, 'uniformoutput', false);
    xlw_d = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xl_d, cos_window, 'uniformoutput', false);
    xlw_r = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xl_r, cos_window, 'uniformoutput', false);
    
    % compute the fourier series
    xlf = cellfun(@fft2, xlw, 'uniformoutput', false);
    xlf_u = cellfun(@fft2, xlw_u, 'uniformoutput', false);  
    xlf_l = cellfun(@fft2, xlw_l, 'uniformoutput', false);
    xlf_d = cellfun(@fft2, xlw_d, 'uniformoutput', false);
    xlf_r = cellfun(@fft2, xlw_r, 'uniformoutput', false);
    %--------------------------------------------------------------------------------------%

    [filter_model_f,spatial_units, channels] = train_filter(xlf, feature_info, yf, seq, params, filter_model_f,reg_window,xlf_u,xlf_l,xlf_d,xlf_r);
    % >
    
    % Update the target size
    % <
    target_sz = base_target_sz(1,:) * currentScaleFactor(1);
    % >
    
    %save position and time
    % <
    tracking_result.center_pos = double(pos);
    tracking_result.target_size = double(target_sz);
    seq = report_tracking_result(seq, tracking_result);
    seq.time = seq.time + toc();
    % >
    
    % visualisation
    % <
    if params.vis_res
        rect_position_vis = [pos([2,1]) - (target_sz([2,1]) - 1)/2, target_sz([2,1])];
        im_to_show = double(im)/255;
        if size(im_to_show,3) == 1
            im_to_show = repmat(im_to_show, [1 1 3]);
        end
        if seq.frame == 1  %first frame, create GUI
            fig_handle = figure('Name', 'Tracking','Position',[100, 300, 600, 480]);
            set(gca, 'position', [0 0 1 1 ]);
            axis off;axis image;
            imagesc(im_to_show);
            hold on;
            rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
            text(10, 10, int2str(seq.frame), 'color', [0 1 1], 'FontSize',20);
            hold off;
            axis off;axis image;set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])
        else
            figure(fig_handle);
            imagesc(im_to_show);
            hold on;
            rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
            text(10, 10, int2str(seq.frame), 'color', [0 1 1], 'FontSize',20);
            hold off;
        end
        drawnow
    end
    
    if params.vis_res && params.vis_details
        if seq.frame == 1
            res_vis_sz = [300, 480];
            fig_handle_detail = figure('Name', 'Details','Position',[700, 300, res_vis_sz]);
            anno_handle = annotation('textbox',[0.3,0.92,0.7,0.08],'LineStyle','none',...
                'String',['Tracking Frame #' num2str(seq.frame)]);
        else
            figure(fig_handle_detail);
            set(anno_handle, 'String',['Tracking Frame #' num2str(seq.frame)]);
            set(gca, 'position', [0 0 1 1 ]);
            subplot('position',[0.05,0.08,0.9,0.12]);
            bar(channels{end}(:)>0);
            xlabel(['channels [' num2str(params.channel_selection_rate(2)) ']']);
            title('Selected Channels for Deep features');
            subplot('position',[0.05,0.3,0.4,0.25]);
            imagesc(spatial_units{1}>0);
            xlabel(['spatial units [' num2str(params.spatial_selection_rate(1)) ']']);
            title('Hand-crafted features');
            subplot('position',[0.55,0.3,0.4,0.25]);
            imagesc(spatial_units{end}>0);
            xlabel(['spatial units [' num2str(params.spatial_selection_rate(2)) ']']);
            title('Deep features');
            subplot('position',[0.05,0.65,0.4,0.25]);
            patch_to_show = imresize(img_samples{1}(:,:,:,sind),res_vis_sz);
            imagesc(patch_to_show);
            hold on;
            sampled_scores_display = circshift(imresize(response(:,:,sind),...
                res_vis_sz),floor(0.5*res_vis_sz));
            resp_handle = imagesc(sampled_scores_display);
            alpha(resp_handle, 0.6);
            hold off;
            axis off;
            title('Response map');
            subplot('position',[0.55,0.65,0.4,0.25]);
            bar([w,1-w]);axis([0.5 2.5 0 1]);
            set(gca,'XTickLabel',{'HC','Deep'});
            title('Weights HC vs. Deep');
        end
    end
   
end
% get tracking results
% <
[~, results,output] = get_sequence_results(seq); 
if params.vis_res&& params.vis_details
    close(fig_handle_detail);close(fig_handle);
elseif params.vis_res
    close(fig_handle);
end
% >

% rank calculation for the entire sequence
% (can be commented on benchmark experiments)
% < %%
if ~isempty(store_filter)
    results.rank_var = rank(store_filter); 
else
    results.rank_var = -1;
end
% > %%

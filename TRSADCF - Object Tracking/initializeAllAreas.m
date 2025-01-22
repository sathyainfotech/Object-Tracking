function [params] = initializeAllAreas(seq,im,video_path, params)
    
%=====================================================================================================================================% 
%params files
params.inner_padding = 0.2;  
params.fixed_area = 160^2;
params.n_bins = 2^5; 
params.grayscale_sequence = true;
params.learning_rate_pwp = 0.04;
params.learning_rate_cf = 0.015;
params.hog_cell_size = 4;
params.feature_type = 'fhog';
params.den_per_channel = false;
params.output_sigma_factor = 1/16 ; 
params.lambda = 1e-3;
params.lambda2 = 0.5;
params.merge_factor = 0.2;
params.merge_method = 'const_factor';
params.visualization = 1;
params.visualization_dbg = 1;
%=====================================================================================================================================%
%% scale related
params.scale_adaptation = true;
params.hog_scale_cell_size = 4;                % Default DSST=4
params.learning_rate_scale = 0.025;
params.scale_sigma_factor = 1/4;
params.num_scales = 33;
params.scale_model_factor = 1.0;
params.scale_step = 1.02;
params.scale_model_max_area = 32*16;
params.fout = -1;
%=====================================================================================================================================%



	avg_dim = sum(params.target_sz)/2;    
	% size from which we extract features
	bg_area = round(params.target_sz + avg_dim);
    
	% pick a "safe" region smaller than bbox to avoid mislabeling
	fg_area = round(params.target_sz - avg_dim * params.inner_padding);
    
	% saturate to image size
	if(bg_area(2)>size(im,2)), bg_area(2)=size(im,2)-1; end
	if(bg_area(1)>size(im,1)), bg_area(1)=size(im,1)-1; end
        
	% make sure the differences are a multiple of 2 (makes things easier later in color histograms)
	bg_area = bg_area - mod(bg_area - params.target_sz, 2);
 
	fg_area = fg_area + mod(bg_area - fg_area, 2);          
	
	area_resize_factor = sqrt(params.fixed_area/prod(bg_area));     
    params.bg_area=bg_area;
    params.fg_area=fg_area;
    params.area_resize_factor=area_resize_factor;
    
	params.norm_bg_area = round(bg_area * area_resize_factor);
    
	params.cf_response_size = floor(params.norm_bg_area / params.hog_cell_size);
	
	norm_target_sz_w = 0.75*params.norm_bg_area(2) - 0.25*params.norm_bg_area(1);
    
 	norm_target_sz_h = 0.75*params.norm_bg_area(1) - 0.25*params.norm_bg_area(2);
	
	params.norm_target_sz = round([norm_target_sz_h norm_target_sz_w]);        
	    
	norm_pad = floor((params.norm_bg_area - params.norm_target_sz) / 2);
    
	radius = min(norm_pad);        
	
	params.norm_delta_area = (2*radius+1) * [1, 1];        
   
	
	params.norm_pwp_search_area = params.norm_target_sz + params.norm_delta_area - 1;
    
    
    
    
    
    %   Grayscale feature parameters
grayscale_params.colorspace='gray';
grayscale_params.nDim = 1;

params.t_global.cell_size = 4;                  % Feature cell size
params.t_global.cell_selection_thresh = 0.75^2; % Threshold for reducing the cell size in low-resolution cases

%   Search region + extended background parameters
params.search_area_shape = 'square';    % the shape of the training/detection window: 'proportional', 'square' or 'fix_padding'
params.search_area_scale = 5;           % the size of the training/detection area proportional to the target size
params.filter_max_area   = 50^2;        % the size of the training/detection area in feature grid cells

%   Learning parameters
params.learning_rate       = 0.013;        % learning rate
params.output_sigma_factor = 1/16;		% standard deviation of the desired correlation output (proportional to target)

%   Detection parameters
params.interpolate_response  = 4;        % correlation score interpolation strategy: 0 - off, 1 - feature grid, 2 - pixel grid, 4 - Newton's method
params.newton_iterations     = 50;           % number of Newton's iteration to maximize the detection scores
				% the weight of the standard (uniform) regularization, only used when params.use_reg_window == 0
%   Scale parameters
params.number_of_scales =  5;
params.scale_step       = 1.01;

params.admm_iterations = 2;
params.admm_lambda = 0.01;

%   Debug and visualization
params.visualization = 1;

%   Global feature parameters 
hog_params.nDim   = 31;
params.t_features = {
    ...struct('getFeature',@get_colorspace, 'fparams',grayscale_params),...  % Grayscale is not used as default
    struct('getFeature',@get_fhog,'fparams',hog_params),...
};

params.video_path = video_path;

%   size, position, frames initialization
params.wsize    = [seq.init_rect(1,4), seq.init_rect(1,3)];

params.init_pos = [seq.init_rect(1,2), seq.init_rect(1,1)] + floor(params.wsize/2);
params.s_frames = seq.s_frames;
params.no_fram = numel(params.s_frames);

%--------------------------------------------------------------------------------------%
%DEEP

% Image sample parameters
params.search_area_shape = 'square';    % The shape of the samples
params.search_area_scale = 4.5;         % The scaling of the target size to get the search area
params.min_image_sample_size = 200^2;   % Minimum area of image samples
params.max_image_sample_size = 250^2;   % Maximum area of image samples

%--------------------------------------------------------------------------------------%

	
end

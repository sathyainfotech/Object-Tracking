function results = run_TRSADCF(seq,im,params)

% Feature specific parameters
hog_params.cell_size = 6;
hog_params.nDim = 31;
hog_params.learning_rate = 0.6;
hog_params.channel_selection_rate = 0.7;
hog_params.feature_is_deep = false;

cn_params.tablename = 'CNnorm';
cn_params.useForGray = false;
cn_params.cell_size = 4;
cn_params.nDim = 10;
cn_params.learning_rate = 0.6;
cn_params.channel_selection_rate = 0.7;
cn_params.feature_is_deep = false;

grayscale_params.colorspace='gray';
grayscale_params.cell_size = 4;
grayscale_params.useForColor = false;
grayscale_params.learning_rate = 0.6;
grayscale_params.channel_selection_rate = 0.7;
grayscale_params.feature_is_deep = false;
params.t_global.cell_size = 4;           

ic_params.tablename = 'intensityChannelNorm6';
ic_params.useForColor = false;
ic_params.cell_size = 4;
ic_params.nDim = 5;
ic_params.learning_rate = 0.6;
ic_params.channel_selection_rate = 0.7;
ic_params.feature_is_deep = false;

cnn_params.nn_name = 'imagenet-resnet-50-dag.mat'; 
cnn_params.output_var = {'res4dx'};    
cnn_params.downsample_factor = 1;           
cnn_params.input_size_mode = 'adaptive';       
cnn_params.input_size_scale = 1;       
cnn_params.learning_rate = 0.06;
cnn_params.channel_selection_rate = 0.07;
cnn_params.feature_is_deep = true;
cnn_params.augment.blur = 1;
cnn_params.augment.rotation = 1;
cnn_params.augment.flip = 1;
%cnn_params.useForGray = false;
    
%struct('getFeature',@get_cnn_layers, 'fparams',cnn_params),...
% Which features to include
params.t_features = {
    struct('getFeature',@get_fhog,'fparams',hog_params),...
    struct('getFeature',@get_table_feature, 'fparams',ic_params),...
    struct('getFeature',@get_table_feature, 'fparams',cn_params),...
    struct('getFeature',@get_cnn_layers, 'fparams',cnn_params),...
};

% Image sample parameters
params.search_area_shape = 'square';    % The shape of the samples
params.search_area_scale = 4.2;         % The scaling of the target size to get the search area
params.min_image_sample_size = 200^2;   % Minimum area of image samples
params.max_image_sample_size = 250^2;   % Maximum area of image samples
params.mask_window_min = 1e-3;           

% Detection parameters
params.refinement_iterations = 1;       % Number of iterations used to refine the resulting position in a frame
params.newton_iterations = 5;           % The number of Newton iterations used for optimizing the detection score
params.clamp_position = false;          % Clamp the target position to be inside the image

% Learning parameters
params.output_sigma_factor = [1/16 1/3];		
params.temporal_consistency_factor = [31 6];
params.stability_factor = 0;

% ADMM parameters
params.max_iterations = 2;
params.init_penalty_factor = 1;
params.penalty_scale_step = 10;

% Scale parameters for the translation model
params.number_of_scales = 7;       
params.scale_step = 1.01;           

% GPU
params.use_gpu = false;              
params.gpu_id = [];               

% Initialize
params.visualization =1;
params.seq = seq;


%SATHYA EDIT
params.fixed_area = 160^2;
params.inner_padding = 0.2;  
params.hog_cell_size = 4;
params.n_bins = 2^5; 

params.merge_factor = 0.2;
params.merge_method = 'const_factor';
params.admm_lambda = 0.01;
avg_dim = sum(params.target_sz)/2;    
bg_area = round(params.target_sz + avg_dim);
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

params.wsize    = [seq.init_rect(1,4), seq.init_rect(1,3)];

% Run tracker
[results] = tracker(params);

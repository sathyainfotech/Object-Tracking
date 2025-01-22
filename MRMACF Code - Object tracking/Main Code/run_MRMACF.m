function [results,output] = run_MRMACF(seq, res_path, bSaveImage)
 
setup_paths();
hog_params.cell_size = 6;           
hog_params.feature_is_deep = false;

% CN feature settings 
cn_params.tablename = 'CNnorm';
cn_params.useForGray = false;
cn_params.cell_size = 4;
cn_params.feature_is_deep = false;       

% IC feature settings
ic_params.tablename = 'intensityChannelNorm6';
ic_params.useForColor = false;
ic_params.cell_size = 4;
ic_params.feature_is_deep = false;

% CNN feature settings
dagnn_params.nn_name = 'imagenet-resnet-50-dag.mat'; 
dagnn_params.output_var = {'res4ex'};    
dagnn_params.feature_is_deep = true;
dagnn_params.augment.blur = 1;
dagnn_params.augment.rotation = 1;
dagnn_params.augment.flip = 1;

params.t_features = {
    struct('getFeature',@get_fhog,'fparams',hog_params),...
    struct('getFeature',@get_table_feature, 'fparams',cn_params),...
    struct('getFeature',@get_dagnn_layers, 'fparams',dagnn_params),...
    struct('getFeature',@get_table_feature, 'fparams',ic_params),...
};

%------------------------------------------------------------------------%
% params.wsize    = [seq.init_rect(1,4), seq.init_rect(1,3)];
% params.init_pos = [seq.init_rect(1,2), seq.init_rect(1,1)] + floor(params.wsize/2);
% 
% params.init_pos_left = params.init_pos - [seq.init_rect(1,4),0];
% params.init_pos_right = params.init_pos + [seq.init_rect(1,4),0];
% params.init_pos_bottom = params.init_pos - [0, seq.init_rect(1,3)];
% params.init_pos_top = params.init_pos + [0, seq.init_rect(1,3)];
% 
% params.init_pos_lefttop = params.init_pos - [seq.init_rect(1,4),-1*seq.init_rect(1,3)];
% params.init_pos_righttop = params.init_pos + [seq.init_rect(1,4),seq.init_rect(1,3)];
% params.init_pos_leftbottom = params.init_pos - [seq.init_rect(1,4), seq.init_rect(1,3)];
% params.init_pos_rightbottom = params.init_pos + [-1*seq.init_rect(1,4), seq.init_rect(1,3)];
% 
% params.width = seq.init_rect(1,4);
% params.height = seq.init_rect(1,3);
% 
% params.keystep = 8;
% params.Period = 2;
% 
% w = params.width;
% h = params.height;
% L = [w h];
% [L_min,~] = min(L);
% cross_l = sqrt(w*w+h*h);
% 
% yta = 0.28;
% params.yta1 = yta * L_min / w;
% params.yta2 = yta * L_min / h;
% params.yta3 = yta * L_min / cross_l;
%------------------------------------------------------------------------%

% Set [non-deep deep] parameters
params.learning_rate = [0.6 0.05];          % updating rate in each learning stage
params.channel_selection_rate = [0.9 0.075];% channel selection ratio
params.spatial_selection_rate = [0.1 0.9];  % spatial units selection ratio
params.output_sigma_factor = [1/16 1/4];	% desired label setting
params.lambda1 = 10;                         % lambda_1
params.lambda2 = 1;                         % lambda_2
params.lambda3 = [16 12];                   % lambda_3
params.stability_factor = [0 0];            % robustness testing parameter

% Image sample parameters
params.search_area_scale = [3.8 4.2];        % search region  
params.min_image_sample_size = [150^2 200^2];% minimal search region size   
params.max_image_sample_size = [200^2 250^2];% maximal search region size  

% Detection parameters
params.refinement_iterations = 1;           % detection numbers
params.newton_iterations = 5;               % subgrid localisation numbers   

% Set scale parameters
params.number_of_scales = 7;                % scale pyramid size
params.scale_step = 1.01;                   % scale step ratio

% Spatial regularization window_parameters
params.feature_downsample_ratio = [4, 14]; %  Feature downsample ratio 
% (We found that decreasing the downsampling ratios of CNN layer may benefit the performance)
params.reg_window_max = 1e5;           % The maximum value of the regularization window
params.reg_window_min = 1e-3;           % The minimum value of the regularization window
params.regularizer_size=1;
params.eta=1e7;

params.offset_prop=1.08;
params.back_imp=0.05;
% params.back_imp=1.5;

%SATHYA
params.sz_ratio = 2.5;
% params.delta = 0.01;
params.delta = 0.001;
params.nu    = 1;
% params.eta   = 0.044;
params.eta   = 0.001;

% Set GPU 
params.use_gpu = true;                
params.gpu_id = [];              

% Initialisation
params.vis_res = 1;                         % visualisation results
params.vis_details = 0;                     % visualisation details for debug
params.seq = seq;   

% Run tracker
[results,output] = tracker_main(params);


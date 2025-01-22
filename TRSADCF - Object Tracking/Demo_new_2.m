 function [precision, fps] = Demo_new_2(video,show_visualization,show_plots)
setup_paths();

% Set your dataset directory path here
base_path  = 'Z:/Users/Owner/Downloads/DATASETS/OTB-2013-All';

	if nargin < 1, video = 'choose'; end
    % if nargin < 1, video = 'all'; end
	if nargin < 2, show_visualization = ~strcmp(video, 'all'); end
	if nargin < 3, show_plots = strcmp(video, 'all'); end

    start_frame=1;

	switch video
	case 'choose'           
        video = choose_video(base_path);        
		if ~isempty(video)
			[precision, fps] = Demo_new_2(video,show_visualization, show_plots);				
		end
		
	case 'all'
		dirs = dir(base_path);
		videos = {dirs.name};
		videos(strcmp('.', videos) | strcmp('..', videos) | strcmp('anno', videos) | ~[dirs.isdir]) = [];
		all_precisions = zeros(numel(videos),1);  %to compute averages
		all_fps = zeros(numel(videos),1);
		
		if ~exist('matlabpool', 'file')
			for k = 1:numel(videos)
				[all_precisions(k), all_fps(k)] = Demo_new_2(videos{k}, show_visualization, show_plots);
			end
		else
			if matlabpool('size') == 0
				matlabpool open;
			end
			parfor k = 1:numel(videos)
				[all_precisions(k), all_fps(k)] = Demo_new_2(videos{k}, show_visualization, show_plots);
			end
        end		
		mean_precision = mean(all_precisions);
		fps = mean(all_fps);
		fprintf('\nAverage precision (20px):% 1.3f, Average FPS:% 4.2f\n\n', mean_precision, fps)
		if nargout > 0
			precision = mean_precision;
		end
		
        otherwise            
            video_path = [base_path '/' video];
            text_files = dir([video_path '*_frames.txt']);

            if(~isempty(text_files))
                f = fopen([video_path text_files(1).name]);
                frames = textscan(f, '%f,%f');        
                fclose(f);
            else
                frames = {};
            end
            if exist('start_frame')
                frames{1} = start_frame;
            else
                frames{1} = 1;
            end
    
            bb_VOT = csvread(fullfile(video_path,'groundtruth_rect.txt'));                        
%           bb_VOT = csvread(fullfile(video_path,'groundtruth.txt'));                        
%           bb_VOT = csvread(fullfile(video_path, 'gt.txt'));
            region = bb_VOT(frames{1},:);
            if(numel(region)==8)
                % polygon format
                [cx, cy, w, h] = getAxisAlignedBB(region);
            else
                x = region(1);
                y = region(2);
                w = region(3);
                h = region(4);
                cx = x+w/2;
                cy = y+h/2;
            end
                
            [seq,ground_truth] = load_video(video_path,video);
           
            % %   size, position, frames initialization
            params.wsize    = [seq.init_rect(1,4), seq.init_rect(1,3)];
            params.init_pos = [seq.init_rect(1,2), seq.init_rect(1,1)] + floor(params.wsize/2);
            params.s_frames = seq.s_frames;
            params.no_fram  = seq.end_frame - seq.start_frame + 1;
            params.seq_st_frame = seq.start_frame;
            params.seq_en_frame = seq.end_frame;                                    
            
            dir_content = dir(fullfile(video_path, 'img'));    
            n_imgs = length(dir_content) - 2;
            img_files = cell(n_imgs, 1);
            for ii = 1:n_imgs
                img_files{ii} = dir_content(ii+2).name;
            end

            img_files(1:start_frame-1)=[];
            img_path = fullfile(video_path, 'img');
            
            im = imread(fullfile(img_path, img_files{1}));    
            
            % init_pos is the centre of the initial bounding box                            
            params.init_pos = [cy cx];
            params.target_sz = round([h w]); 
            [params] = initializeAllAreas(seq,im,video_path, params);            
            
            if(size(im,3)==1)
                params.grayscale_sequence = false;
                 disp(params.grayscale_sequence);
            else
                params.grayscale_sequence = false;
            end
            params.img_files = img_files;
            params.img_path = img_path;                                             
            
            results = run_TRSADCF(seq,im,params);
            precisions = precision_plot(results.res,bb_VOT,video,show_plots);                      
            fps = results.fps;
            fprintf('%12s - Precision (20px):% 1.3f, FPS:% 4.2f\n', video, precisions(20), fps);
             
       q1= results.res;
       clear results
       results{1}.type = 'rect';
       x =q1(:,2);
       results{1}.res = q1;
       results{1}.startFame = 1;
       results{1}.annoBegin = 1;
       results{1}.len = size(x, 1);

       % Set your folder name to save your results
       res_path = ['Set Your Folder Name Here'  '/'];
       if ~isfolder(res_path)
           mkdir(res_path);
       end

       % Set your tracker name instead of TRSADCF (Optional)
       save([res_path lower(video) '_TRSADCF.mat'], 'results');           
                if nargout > 0
                    precision = precisions(20);
                end
	end
end

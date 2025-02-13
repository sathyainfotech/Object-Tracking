function [seq, results,output] = get_sequence_results(seq)

if strcmpi(seq.format, 'otb')||strcmpi(seq.format, 'otb_8')
    results.type = 'rect';
    results.res = seq.rect_position;
    output = seq.rect_position;
elseif strcmpi(seq.format, 'vot')
    seq.handle.quit(seq.handle);
else
    error('Uknown sequence format');
end

if isfield(seq, 'time')
    results.fps = seq.num_frames / seq.time;
else
    results.fps = NaN;
end
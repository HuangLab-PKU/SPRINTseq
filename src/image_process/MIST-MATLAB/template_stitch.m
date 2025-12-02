function template_stitch(src_dir, dest_dir, tile_x, tile_y)
    log_file_path = fullfile(dest_dir, 'img-debug.log');
    src_im_name_pattern = 'FocalStack_{ppp}.tif';
    [im_name_grid, to_stitch_time_slice_nbs] = build_img_name_grid(src_dir, src_im_name_pattern, tile_x, tile_y, ...
        'time_slices_to_stitch', 'all', ...
        'tiling_technique', 'combing', ...
        'starting_point', 'upperright', ...
        'first_direction', 'horizontal');

    alpha = 1.5;
    assemble_from_metadata = false;
    output_prefix = 'img-';
    blend_method = 'Overlay';
    estimated_overlap_x = NaN;
    estimated_overlap_y = NaN;
    percent_overlap_error = NaN;
    repeatability = NaN;
    save_stitched_image = 1;
    time_slice = 1;

    stitch_time_slice(src_dir, im_name_grid, dest_dir, output_prefix, time_slice, repeatability, ...
        percent_overlap_error, blend_method, alpha, save_stitched_image, assemble_from_metadata, log_file_path, estimated_overlap_x, ...
        estimated_overlap_y);

end

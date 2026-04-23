"""
Gene mapping methods for ISS data.

This module provides different methods to map intensity data to genes:
- threshold: Fixed or adaptive threshold-based base calling + Hamming distance matching
- intensity_direct: Direct intensity similarity matching (starfish MetricDistance-style)
- per_round_max: Per-round max channel decoding (starfish PerRoundMaxChannel-style)
- probabilistic: Probabilistic base calling (to be implemented)
"""

import pandas as pd
import numpy as np

from .sequence_generation import get_seq_df
from .reference_check import check_sequence
from .mapping import map_barcode, unstack_plex
from tqdm import tqdm


def compute_adaptive_thresholds(intensity_df, cyc_num=10, channels=['cy3', 'cy5'],
                                percentile=10, min_threshold=20, max_threshold=200,
                                verbose=True):
    """
    Compute adaptive thresholds for each cycle and channel based on intensity distribution.
    
    This helps handle signal decay across cycles and varying signal strengths.
    
    Parameters
    ----------
    intensity_df : pd.DataFrame
        DataFrame with intensity columns
    cyc_num : int
        Number of cycles (default: 10)
    channels : list
        List of channel names (default: ['cy3', 'cy5'])
    percentile : float
        Percentile to use as threshold (default: 10, meaning 10th percentile)
    min_threshold : float
        Minimum threshold value (default: 20)
    max_threshold : float
        Maximum threshold value (default: 200)
    verbose : bool
        Print progress (default: True)
    
    Returns
    -------
    dict
        Dictionary with keys like 'cyc_1_cy3', 'cyc_1_cy5', etc., mapping to threshold values
    """
    thresholds = {}
    
    if verbose:
        print("  Computing adaptive thresholds per cycle and channel...")
    
    for cyc in range(1, cyc_num + 1):
        for channel in channels:
            col = f'cyc_{cyc}_{channel}'
            if col not in intensity_df.columns:
                # Use default if column missing
                thresholds[col] = min_threshold
                continue
            
            intensities = intensity_df[col].values
            intensities = intensities[~np.isnan(intensities)]  # Remove NaN
            intensities = intensities[intensities > 0]  # Remove zeros
            
            if len(intensities) == 0:
                thresholds[col] = min_threshold
                continue
            
            # Compute percentile threshold
            threshold = np.percentile(intensities, percentile)
            
            # Clip to min/max bounds
            threshold = max(min_threshold, min(max_threshold, threshold))
            
            thresholds[col] = float(threshold)
    
    if verbose:
        # Print summary
        avg_thresholds = {}
        for channel in channels:
            channel_thresholds = [thresholds.get(f'cyc_{cyc}_{channel}', min_threshold) 
                                 for cyc in range(1, cyc_num + 1)]
            avg_thresholds[channel] = np.mean(channel_thresholds)
        print(f"  Average thresholds: {avg_thresholds}")
    
    return thresholds


def threshold_mapping(intensity_df, ref_file, thresholds=None, cyc_num=10, 
                      channels=['cy3', 'cy5'], hamming_max=1, exact_match=False,
                      check_sequence_first=True, adaptive_threshold=False,
                      threshold_percentile=10, min_threshold=20, max_threshold=200,
                      verbose=True):
    """
    Map genes using threshold-based base calling and Hamming distance matching.
    
    This is the standard method that:
    1. Applies thresholds to intensity values (fixed or adaptive)
    2. Generates sequences by mapping boolean pairs to bases
    3. Matches sequences to reference using Hamming distance
    4. Maps matched sequences to genes
    
    Parameters
    ----------
    intensity_df : pd.DataFrame
        DataFrame with columns 'Y', 'X' and 'cyc_{n}_{channel}' for each cycle and channel.
        Must have 'index' column for output matching.
    ref_file : str or Path
        Path to reference file (CSV with Barcode and Gene columns)
    thresholds : dict, optional
        Dictionary mapping channel to threshold value (for fixed thresholds).
        Default: {'cy3': 50, 'cy5': 50}
        If adaptive_threshold=True, this is ignored.
    cyc_num : int
        Number of sequencing cycles (default: 10)
    channels : list
        List of channel names (default: ['cy3', 'cy5'])
    hamming_max : int
        Maximum allowed Hamming distance for matching (default: 1)
    exact_match : bool
        If True, require exact match (hamming_max=0) (default: False)
    check_sequence_first : bool
        If True, run check_sequence before mapping (default: True)
    adaptive_threshold : bool
        If True, compute adaptive thresholds per cycle/channel (default: False)
    threshold_percentile : float
        Percentile to use for adaptive thresholds (default: 10)
    min_threshold : float
        Minimum threshold value for adaptive thresholds (default: 20)
    max_threshold : float
        Maximum threshold value for adaptive thresholds (default: 200)
    verbose : bool
        If True, print progress information (default: True)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ['index', 'Sequence', 'Gene']
        - index: matches input intensity_df index
        - Sequence: generated sequence string
        - Gene: mapped gene name (NaN if not matched)
    """
    # Compute or use provided thresholds
    if adaptive_threshold:
        adaptive_thresholds = compute_adaptive_thresholds(
            intensity_df, cyc_num, channels, threshold_percentile,
            min_threshold, max_threshold, verbose
        )
        # Convert to per-channel format for get_seq_df (it uses channel-level thresholds)
        # We'll use the average threshold per channel across cycles
        thresholds = {}
        for channel in channels:
            channel_thresholds = [adaptive_thresholds.get(f'cyc_{cyc}_{channel}', min_threshold)
                                 for cyc in range(1, cyc_num + 1)]
            thresholds[channel] = np.mean(channel_thresholds)
    elif thresholds is None:
        thresholds = {'cy3': 50, 'cy5': 50}
    
    if exact_match:
        hamming_max = 0
    
    # Step 1: Generate sequences from intensity data
    if verbose:
        print("  Generating sequences from intensity data...")
    seq_df = get_seq_df(intensity_df, thresholds, cyc_num=cyc_num, channels=channels)
    
    # Add index column for matching
    if 'index' not in seq_df.columns:
        seq_df['index'] = intensity_df['index'].values if 'index' in intensity_df.columns else intensity_df.index.values
    
    # Step 2: Check sequences (optional, for quality control)
    if check_sequence_first:
        if verbose:
            print("  Checking sequences against reference...")
        # Create temporary file for check_sequence
        import tempfile
        import os
        tmp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        tmp_path = tmp_file.name
        seq_df[['Y', 'X', 'Sequence']].to_csv(tmp_path, index=False)
        tmp_file.close()
        
        try:
            checked_df = check_sequence(tmp_path, str(ref_file), verbose=verbose)
            # Merge checked sequences back to original seq_df using Y, X coordinates
            # This preserves the index mapping and only updates sequences that passed the check
            checked_df = checked_df[['Y', 'X', 'Sequence']].copy()
            # Ensure Y, X are int64 for proper merging
            checked_df['Y'] = checked_df['Y'].astype('int64')
            checked_df['X'] = checked_df['X'].astype('int64')
            seq_df['Y'] = seq_df['Y'].astype('int64')
            seq_df['X'] = seq_df['X'].astype('int64')
            
            # Merge: keep all original rows, but update Sequence for rows that passed check
            seq_df = seq_df.merge(
                checked_df,
                on=['Y', 'X'],
                how='left',
                suffixes=('', '_checked')
            )
            # Use checked sequence if available, otherwise keep original
            seq_df['Sequence'] = seq_df['Sequence_checked'].fillna(seq_df['Sequence'])
            seq_df = seq_df[['index', 'Y', 'X', 'Sequence']].copy()
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    # Step 3: Map barcodes to genes
    if verbose:
        print("  Mapping sequences to genes...")
    
    # Create temporary file for map_barcode
    import tempfile
    import os
    tmp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    tmp_path = tmp_file.name
    seq_df[['Y', 'X', 'Sequence']].to_csv(tmp_path, index=False)
    tmp_file.close()
    
    try:
        map_df = map_barcode(tmp_path, str(ref_file), exact=exact_match)
        map_df = unstack_plex(map_df)
        
        # Merge mapping results
        map_df_merged = map_df[['Y', 'X', 'Gene']].copy()
        # Ensure Y, X are int64 for proper merging
        map_df_merged['Y'] = map_df_merged['Y'].astype('int64')
        map_df_merged['X'] = map_df_merged['X'].astype('int64')
        
        # Merge with sequence dataframe
        result_df = seq_df.merge(
            map_df_merged,
            on=['Y', 'X'],
            how='left'
        )
        
        # Select and reorder columns
        result_df = result_df[['index', 'Sequence', 'Gene']].copy()
        
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    
    if verbose:
        mapped_count = result_df['Gene'].notna().sum()
        print(f"  Mapped {mapped_count}/{len(result_df)} points to genes")
    
    return result_df


def build_codebook_intensity_patterns(ref_file, cyc_num=10, channels=['cy3', 'cy5'],
                                       base_intensity=100, noise_level=10, verbose=True):
    """
    Build expected intensity patterns from codebook sequences (starfish-style).
    
    This function converts DNA barcodes to expected intensity patterns based on
    the base-to-channel mapping: A=(cy3,cy5)=(1,1), T=(1,0), C=(0,1), G=(0,0).
    
    Parameters
    ----------
    ref_file : str or Path
        Path to reference file (CSV with Barcode and Gene columns)
    cyc_num : int
        Number of cycles (default: 10)
    channels : list
        Channel names (default: ['cy3', 'cy5'])
    base_intensity : float
        Base intensity value for "on" channels (default: 100)
    noise_level : float
        Noise level for "off" channels (default: 10)
    verbose : bool
        Print progress (default: True)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ['Barcode', 'Gene', 'intensity_pattern']
        where intensity_pattern is a numpy array of shape (cyc_num * len(channels),)
    """
    ref_df = pd.read_csv(ref_file)
    
    # Handle different column name formats
    if 'Barcode' in ref_df.columns and 'Gene' in ref_df.columns:
        barcode_col = 'Barcode'
        gene_col = 'Gene'
    elif 'barcode' in ref_df.columns and 'gene' in ref_df.columns:
        barcode_col = 'barcode'
        gene_col = 'gene'
    elif len(ref_df.columns) >= 2:
        # Assume first column is gene, second is barcode
        gene_col = ref_df.columns[0]
        barcode_col = ref_df.columns[1]
    else:
        raise ValueError(f"Cannot determine column names in reference file: {ref_file}")
    
    # Base to channel mapping: (cy3, cy5)
    base_to_channels = {
        'A': (True, True),   # Both channels on
        'T': (True, False),  # Only cy3 on
        'C': (False, True),   # Only cy5 on
        'G': (False, False)   # Both channels off
    }
    
    patterns = []
    barcodes = []
    genes = []
    
    if verbose:
        print(f"  Building intensity patterns for {len(ref_df)} barcodes...")
    
    for _, row in tqdm(ref_df.iterrows(), total=len(ref_df), desc='  Building patterns', disable=not verbose):
        barcode = str(row[barcode_col]).upper()
        gene = row[gene_col]
        
        # Build intensity pattern for this barcode
        pattern = []
        for cyc in range(1, cyc_num + 1):
            if cyc <= len(barcode):
                base = barcode[cyc - 1]
                cy3_on, cy5_on = base_to_channels.get(base, (False, False))
            else:
                # If barcode is shorter than cyc_num, treat as off
                cy3_on, cy5_on = (False, False)
            
            # Add intensity for each channel
            for channel in channels:
                if channel == 'cy3':
                    intensity = base_intensity if cy3_on else noise_level
                elif channel == 'cy5':
                    intensity = base_intensity if cy5_on else noise_level
                else:
                    intensity = noise_level
                pattern.append(intensity)
        
        patterns.append(np.array(pattern))
        barcodes.append(barcode)
        genes.append(gene)
    
    result_df = pd.DataFrame({
        'Barcode': barcodes,
        'Gene': genes,
        'intensity_pattern': patterns
    })
    
    return result_df


def compute_similarity_matrix(intensity_vectors, codebook_patterns, metric='cosine'):
    """
    Compute similarity matrix between all intensity vectors and all codebook patterns using vectorized operations.
    
    This is much faster than computing similarities one by one.
    
    Parameters
    ----------
    intensity_vectors : np.ndarray
        Array of shape (n_points, n_features) - all intensity vectors
    codebook_patterns : np.ndarray
        Array of shape (n_patterns, n_features) - all codebook patterns
    metric : str
        Similarity metric: 'cosine', 'euclidean', 'pearson', or 'metric_distance'
        
    Returns
    -------
    np.ndarray
        Similarity matrix of shape (n_points, n_patterns)
        similarity_matrix[i, j] = similarity between intensity_vectors[i] and codebook_patterns[j]
    """
    n_points = intensity_vectors.shape[0]
    n_patterns = codebook_patterns.shape[0]
    
    if metric == 'cosine':
        # Vectorized cosine similarity: (intensity_vectors @ codebook_patterns.T) / (norms)
        # intensity_vectors: (n_points, n_features)
        # codebook_patterns: (n_patterns, n_features)
        # Result: (n_points, n_patterns)
        
        # Compute dot products: (n_points, n_patterns)
        dot_products = intensity_vectors @ codebook_patterns.T
        
        # Compute norms
        intensity_norms = np.linalg.norm(intensity_vectors, axis=1, keepdims=True)  # (n_points, 1)
        pattern_norms = np.linalg.norm(codebook_patterns, axis=1)  # (n_patterns,)
        
        # Avoid division by zero
        intensity_norms[intensity_norms == 0] = 1.0
        pattern_norms[pattern_norms == 0] = 1.0
        
        # Broadcast: (n_points, 1) * (1, n_patterns) = (n_points, n_patterns)
        similarity_matrix = dot_products / (intensity_norms * pattern_norms[None, :])
        
        return similarity_matrix
    
    elif metric == 'euclidean':
        # Vectorized euclidean distance
        # Compute squared distances: (n_points, n_patterns)
        # Using: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a*b
        intensity_norms_sq = np.sum(intensity_vectors ** 2, axis=1, keepdims=True)  # (n_points, 1)
        pattern_norms_sq = np.sum(codebook_patterns ** 2, axis=1)  # (n_patterns,)
        dot_products = intensity_vectors @ codebook_patterns.T  # (n_points, n_patterns)
        
        squared_distances = intensity_norms_sq + pattern_norms_sq[None, :] - 2 * dot_products
        distances = np.sqrt(np.maximum(squared_distances, 0))  # Avoid negative due to numerical errors
        
        # Convert to similarity
        max_distances = intensity_norms_sq ** 0.5 + pattern_norms_sq[None, :] ** 0.5
        max_distances[max_distances == 0] = 1.0
        similarity_matrix = 1.0 - (distances / max_distances)
        
        return similarity_matrix
    
    elif metric == 'pearson':
        # Vectorized Pearson correlation
        # Center the vectors
        intensity_centered = intensity_vectors - np.mean(intensity_vectors, axis=1, keepdims=True)
        pattern_centered = codebook_patterns - np.mean(codebook_patterns, axis=1, keepdims=True)
        
        # Compute numerator: dot products of centered vectors
        numerator = intensity_centered @ pattern_centered.T  # (n_points, n_patterns)
        
        # Compute denominators: norms of centered vectors
        intensity_norms = np.linalg.norm(intensity_centered, axis=1, keepdims=True)  # (n_points, 1)
        pattern_norms = np.linalg.norm(pattern_centered, axis=1)  # (n_patterns,)
        
        # Avoid division by zero
        intensity_norms[intensity_norms == 0] = 1.0
        pattern_norms[pattern_norms == 0] = 1.0
        
        similarity_matrix = numerator / (intensity_norms * pattern_norms[None, :])
        
        return similarity_matrix
    
    elif metric == 'metric_distance':
        # Vectorized MetricDistance (starfish-style, more orthodox version)
        #
        # Important:
        #   - Here we internally L2-normalize BOTH intensity vectors and codebook patterns.
        #   - After normalization, all vectors lie on the unit sphere and distance primarily
        #     reflects pattern shape differences, not absolute brightness.
        #   - This makes the metric much less sensitive to global intensity scale and better
        #     suited for comparing decoded patterns, which matches your preference to avoid
        #     false positives caused purely by brightness differences.
        #
        # Steps:
        #   1. L2-normalize intensity_vectors and codebook_patterns
        #   2. Compute Euclidean distance on the unit sphere
        #   3. Convert distance to similarity in (0, 1], where 1 == perfect match

        # 1) L2-normalize intensity vectors
        intensity_norms = np.linalg.norm(intensity_vectors, axis=1, keepdims=True)  # (n_points, 1)
        intensity_norms[intensity_norms == 0] = 1.0  # Avoid division by zero
        intensity_unit = intensity_vectors / intensity_norms

        # 2) L2-normalize codebook patterns
        pattern_norms = np.linalg.norm(codebook_patterns, axis=1, keepdims=True)  # (n_patterns, 1)
        pattern_norms[pattern_norms == 0] = 1.0
        patterns_unit = codebook_patterns / pattern_norms  # (n_patterns, n_features)

        # 3) Compute squared distances on the unit sphere
        #    ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a·b
        #    For unit vectors, ||a|| = ||b|| = 1, so:
        #      ||a - b||^2 = 2 - 2 * (a·b)
        dot_products = intensity_unit @ patterns_unit.T  # (n_points, n_patterns)
        # Clamp dot products to [-1, 1] to avoid numerical issues
        dot_products = np.clip(dot_products, -1.0, 1.0)
        squared_distances = 2.0 - 2.0 * dot_products
        distances = np.sqrt(np.maximum(squared_distances, 0.0))

        # 4) Convert distance to similarity: 1 / (1 + distance)
        #    - distance = 0   -> similarity = 1
        #    - distance large -> similarity -> 0
        similarity_matrix = 1.0 / (1.0 + distances)

        return similarity_matrix
    
    else:
        raise ValueError(f"Unknown similarity metric: {metric}")


def intensity_direct_mapping(intensity_df, ref_file, similarity_metric='metric_distance',
                             top_k=3, min_similarity=0, cyc_num=10, channels=['cy3', 'cy5'],
                             base_intensity=100, noise_level=10,
                             batch_size=100000, verbose=True):
    """
    Map genes using direct intensity similarity matching (starfish-style).
    
    This method directly compares intensity vectors to reference intensity patterns
    without generating sequences first. It mimics starfish's MetricDistance decoder.
    
    Parameters
    ----------
    intensity_df : pd.DataFrame
        DataFrame with intensity columns and 'index' column
    ref_file : str or Path
        Path to reference file (CSV with Barcode and Gene columns)
    similarity_metric : str
        Similarity metric: 'cosine', 'euclidean', 'pearson', or 'metric_distance' (default: 'metric_distance')
    top_k : int
        Return top-k most similar genes (default: 3)
    min_similarity : float
        Minimum similarity threshold (default: 0)
    cyc_num : int
        Number of cycles (default: 10)
    channels : list
        Channel names (default: ['cy3', 'cy5'])
    base_intensity : float
        Base intensity for building codebook patterns (default: 100)
    noise_level : float
        Noise level for building codebook patterns (default: 10)
    batch_size : int
        Batch size for processing points to avoid memory issues (default: 100000)
        If None, processes all points at once (may cause memory error for large datasets)
    verbose : bool
        Print progress (default: True)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ['index', 'Sequence', 'Gene', 'similarity']
        - index: matches input intensity_df index
        - Sequence: barcode sequence (from codebook)
        - Gene: mapped gene name (NaN if not matched)
        - similarity: similarity score (NaN if not matched)
    """
    # Step 1: Build codebook intensity patterns
    if verbose:
        print("  Building codebook intensity patterns...")
    codebook_df = build_codebook_intensity_patterns(
        ref_file, cyc_num, channels, base_intensity, noise_level, verbose
    )
    
    # Step 2: Extract intensity vectors from input dataframe
    if verbose:
        print("  Extracting intensity vectors...")
    intensity_cols = [f'cyc_{cyc}_{ch}' for cyc in range(1, cyc_num + 1) for ch in channels]
    
    # Check if all required columns exist
    missing_cols = [col for col in intensity_cols if col not in intensity_df.columns]
    if missing_cols:
        raise ValueError(f"Missing intensity columns: {missing_cols}")
    
    # Build intensity vectors
    intensity_vectors = intensity_df[intensity_cols].values.astype(float)
    
    # Step 3: Match intensity vectors to codebook patterns using vectorized operations (with batching)
    if verbose:
        print("  Matching intensity vectors to codebook (vectorized with batching)...")
    
    # Convert codebook patterns to numpy array for efficient computation
    codebook_patterns = np.array([p for p in codebook_df['intensity_pattern'].values])
    # Extract Barcode and Gene arrays before deleting codebook_df (needed later for results)
    codebook_barcodes = codebook_df['Barcode'].values
    codebook_genes = codebook_df['Gene'].values
    # Free memory: delete codebook_df after extracting needed arrays
    del codebook_df
    import gc
    gc.collect()
    
    n_points = len(intensity_vectors)
    n_patterns = len(codebook_patterns)
    
    # Estimate memory needed for processing all points at once (in GB)
    estimated_memory_gb = (n_points * n_patterns * 8) / (1024**3)
    
    # Determine batch size if not provided or if it would cause memory issues
    if batch_size is None:
        # If estimated memory is large, reduce batch size to keep memory under target
        target_memory_gb = 5.0
        if estimated_memory_gb > target_memory_gb:
            batch_size = max(10000, int(n_points * target_memory_gb / estimated_memory_gb))
            if verbose:
                print(f"  Auto-detected batch_size={batch_size} to keep memory under {target_memory_gb}GB per batch")
        else:
            # Safe to process all points in a single batch
            batch_size = n_points
            if verbose:
                print(f"  Estimated memory ~{estimated_memory_gb:.2f}GB; processing all {n_points} points in a single batch")
    else:
        # User-specified batch_size; just report estimated memory for full matrix for reference
        if verbose:
            print(f"  Using user-specified batch_size={batch_size} for {n_points} points "
                  f"(full matrix would require ~{estimated_memory_gb:.2f}GB)")
    
    # Initialize result lists (will store multiple matches per point if top_k > 1)
    result_rows = []
    
    # Process in batches
    n_batches = (n_points + batch_size - 1) // batch_size
    
    if verbose:
        print(f"  Processing {n_points} points in {n_batches} batches (batch_size={batch_size})...")
        print(f"  Each batch: ~{batch_size} points × {n_patterns} patterns")
        print(f"  Returning top-{top_k} matches per point (min_similarity={min_similarity})")
    
    # Get index column for result
    index_values = intensity_df['index'].values if 'index' in intensity_df.columns else intensity_df.index.values
    
    for batch_idx in tqdm(range(n_batches), desc='  Processing batches', disable=not verbose):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_points)
        batch_vectors = intensity_vectors[start_idx:end_idx]
        batch_size_actual = len(batch_vectors)
        
        # Compute similarity matrix for this batch: (batch_size, n_patterns)
        batch_similarity_matrix = compute_similarity_matrix(
            batch_vectors, codebook_patterns, similarity_metric
        )
        
        # Free memory: delete batch_vectors after computing similarity (no longer needed)
        del batch_vectors
        
        # Find top-k matches for this batch
        # All metrics are converted to similarity (higher is better) in compute_similarity_matrix
        if top_k == 1:
            # Simple case: just find the best match
            batch_topk_indices = np.argmax(batch_similarity_matrix, axis=1, keepdims=True)  # (batch_size, 1)
            batch_topk_similarities = np.max(batch_similarity_matrix, axis=1, keepdims=True)  # (batch_size, 1)
        else:
            # Find top-k: get indices of top-k largest values
            # Use argpartition for efficiency (faster than full sort when k << n_patterns)
            # argpartition with -matrix gives indices of top-k largest values
            batch_topk_indices = np.argpartition(-batch_similarity_matrix, top_k - 1, axis=1)[:, :top_k]
            
            # Sort within top-k to get descending order (highest similarity first)
            # Get the top-k similarities for sorting
            topk_similarities = batch_similarity_matrix[
                np.arange(batch_size_actual)[:, None], batch_topk_indices
            ]
            # Sort indices by similarity (descending)
            sort_order = np.argsort(-topk_similarities, axis=1)
            batch_topk_indices = batch_topk_indices[
                np.arange(batch_size_actual)[:, None], sort_order
            ]
            batch_topk_similarities = batch_similarity_matrix[
                np.arange(batch_size_actual)[:, None], batch_topk_indices
            ]
        
        # Process each point in the batch
        for i in range(batch_size_actual):
            global_idx = start_idx + i
            point_index = index_values[global_idx]
            
            # Get top-k matches for this point
            point_topk_indices = batch_topk_indices[i]  # (top_k,)
            point_topk_similarities = batch_topk_similarities[i]  # (top_k,)
            
            # Apply threshold and filter
            valid_mask = point_topk_similarities >= min_similarity
            
            if np.any(valid_mask):
                # Add valid matches to results
                valid_indices = point_topk_indices[valid_mask]
                valid_similarities = point_topk_similarities[valid_mask]
                
                # Add matches in order (highest similarity first)
                for rank, (match_idx, similarity) in enumerate(zip(valid_indices, valid_similarities), start=1):
                    result_rows.append({
                        'index': point_index,
                        'Sequence': codebook_barcodes[match_idx],
                        'Gene': codebook_genes[match_idx],
                        'similarity': similarity,
                        'rank': rank
                    })
        
        # Free memory: delete batch similarity matrix after processing
        del batch_similarity_matrix, batch_topk_indices, batch_topk_similarities
    
    # Step 4: Create result dataframe in wide format (one row per point)
    # Each point will have columns: Sequence, Gene, similarity, Sequence.1, Gene.1, similarity.1, ...
    
    # Get all unique point indices
    all_indices = index_values
    n_points = len(all_indices)
    
    # Create index mapping for fast lookup
    index_to_pos = {idx: pos for pos, idx in enumerate(all_indices)}
    
    # Initialize result dataframe with all points
    result_dict = {'index': all_indices}
    
    # Initialize columns for top-k results
    for k in range(1, top_k + 1):
        if k == 1:
            result_dict['Sequence'] = [np.nan] * n_points
            result_dict['Gene'] = [np.nan] * n_points
            result_dict['similarity'] = [np.nan] * n_points
        else:
            result_dict[f'Sequence.{k-1}'] = [np.nan] * n_points
            result_dict[f'Gene.{k-1}'] = [np.nan] * n_points
            result_dict[f'similarity.{k-1}'] = [np.nan] * n_points
    
    # Fill in results from result_rows
    if len(result_rows) > 0:
        # Group by index and sort by rank
        from collections import defaultdict
        point_results_dict = defaultdict(list)
        
        for row in result_rows:
            point_results_dict[row['index']].append(row)
        
        # Fill in the result dictionary
        for idx, matches in point_results_dict.items():
            # Sort matches by rank
            matches_sorted = sorted(matches, key=lambda x: x['rank'])
            pos = index_to_pos[idx]
            
            for k, match in enumerate(matches_sorted[:top_k], start=1):
                if k == 1:
                    result_dict['Sequence'][pos] = match['Sequence']
                    result_dict['Gene'][pos] = match['Gene']
                    result_dict['similarity'][pos] = match['similarity']
                else:
                    result_dict[f'Sequence.{k-1}'][pos] = match['Sequence']
                    result_dict[f'Gene.{k-1}'][pos] = match['Gene']
                    result_dict[f'similarity.{k-1}'][pos] = match['similarity']
    
    # Create DataFrame
    result_df = pd.DataFrame(result_dict)
    
    if verbose:
        mapped_count = result_df['Gene'].notna().sum()
        print(f"  Mapped {mapped_count}/{len(result_df)} points to genes "
              f"({mapped_count/len(result_df)*100:.1f}%)")
        if mapped_count > 0:
            avg_similarity = result_df['similarity'].mean()
            print(f"  Average similarity: {avg_similarity:.3f}")
    
    return result_df


def per_round_max_channel_mapping(intensity_df, ref_file, cyc_num=10, 
                                   channels=['cy3', 'cy5'], min_confidence=0.5,
                                   channel_balance=None, auto_balance=True,
                             verbose=True):
    """
    Map genes using PerRoundMaxChannel method (starfish-style).
    
    This method selects the channel with maximum intensity in each round,
    then matches the resulting pattern to the codebook.
    
    **Important**: This method assumes channels are balanced. If cy3 and cy5 have
    systematic brightness differences, use channel_balance or auto_balance to correct.
    
    Parameters
    ----------
    intensity_df : pd.DataFrame
        DataFrame with intensity columns and 'index' column
    ref_file : str or Path
        Path to reference file (CSV with Barcode and Gene columns)
    cyc_num : int
        Number of cycles (default: 10)
    channels : list
        Channel names (default: ['cy3', 'cy5'])
    min_confidence : float
        Minimum confidence threshold (default: 0.5)
    channel_balance : dict, optional
        Manual channel balance factors: {'cy3': factor_cy3, 'cy5': factor_cy5}
        If provided, intensities will be multiplied by these factors before decoding.
        If None and auto_balance=True, will be computed automatically.
    auto_balance : bool
        If True, automatically compute channel balance factors from data (default: True)
        This estimates the ratio between channels when both should be "on" (base A).
    verbose : bool
        Print progress (default: True)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ['index', 'Sequence', 'Gene', 'confidence']
    """
    # Step 1: Load codebook
    if verbose:
        print("  Loading codebook...")
    ref_df = pd.read_csv(ref_file)
    
    # Handle different column name formats
    if 'Barcode' in ref_df.columns and 'Gene' in ref_df.columns:
        barcode_col = 'Barcode'
        gene_col = 'Gene'
    elif 'barcode' in ref_df.columns and 'gene' in ref_df.columns:
        barcode_col = 'barcode'
        gene_col = 'gene'
    elif len(ref_df.columns) >= 2:
        gene_col = ref_df.columns[0]
        barcode_col = ref_df.columns[1]
    else:
        raise ValueError(f"Cannot determine column names in reference file: {ref_file}")
    
    # Create mapping: barcode -> gene
    barcode_to_gene = dict(zip(ref_df[barcode_col].astype(str).str.upper(), ref_df[gene_col]))
    
    # Step 1.5: Compute or use channel balance factors
    intensity_cols = [f'cyc_{cyc}_{ch}' for cyc in range(1, cyc_num + 1) for ch in channels]
    missing_cols = [col for col in intensity_cols if col not in intensity_df.columns]
    if missing_cols:
        raise ValueError(f"Missing intensity columns: {missing_cols}")
    
    if channel_balance is None and auto_balance:
        if verbose:
            print("  Computing channel balance factors...")
            print("  Note: Assuming data has NOT been balanced in intensity correction step.")
            print("  If data was already balanced, set auto_balance=False and channel_balance=None")
        
        # Improved method: Use T and C patterns instead of A (since encoding may not have A)
        # T = (cy3 high, cy5 low), C = (cy3 low, cy5 high)
        # We can estimate balance by comparing T and C intensities
        
        from .sequence_generation import get_seq_df
        
        # Use a low threshold to get many candidates
        thresholds = {}
        for ch in channels:
            all_values = np.concatenate([intensity_df[f'cyc_{cyc}_{ch}'].values 
                                        for cyc in range(1, cyc_num + 1)])
            thresholds[ch] = np.percentile(all_values, 10)
        
        # Generate sequences to identify T and C positions
        seq_df = get_seq_df(intensity_df, thresholds, cyc_num=cyc_num, channels=channels)
        
        cy3_values_t = []  # cy3 intensities at T positions (should be high)
        cy5_values_c = []  # cy5 intensities at C positions (should be high)
        
        for idx, row in seq_df.iterrows():
            seq = row['Sequence']
            for cyc in range(1, min(cyc_num + 1, len(seq) + 1)):
                base = seq[cyc - 1]
                cy3_col = f'cyc_{cyc}_cy3'
                cy5_col = f'cyc_{cyc}_cy5'
                
                if base == 'T' and cy3_col in intensity_df.columns:
                    cy3_values_t.append(intensity_df.iloc[idx][cy3_col])
                elif base == 'C' and cy5_col in intensity_df.columns:
                    cy5_values_c.append(intensity_df.iloc[idx][cy5_col])
        
        if len(cy3_values_t) > 100 and len(cy5_values_c) > 100:
            # Estimate balance: T and C should have similar intensities when "on"
            cy3_median_t = np.median(cy3_values_t)
            cy5_median_c = np.median(cy5_values_c)
            
            if cy3_median_t > 0 and cy5_median_c > 0:
                # Balance factors: make T and C have similar "on" intensities
                if cy3_median_t > cy5_median_c:
                    balance_cy3 = cy5_median_c / cy3_median_t
                    balance_cy5 = 1.0
                else:
                    balance_cy3 = 1.0
                    balance_cy5 = cy3_median_t / cy5_median_c
                
                channel_balance = {'cy3': balance_cy3, 'cy5': balance_cy5}
                
                if verbose:
                    print(f"  Channel balance factors: cy3={balance_cy3:.3f}, cy5={balance_cy5:.3f}")
                    print(f"  (T median cy3={cy3_median_t:.1f}, C median cy5={cy5_median_c:.1f})")
            else:
                channel_balance = {'cy3': 1.0, 'cy5': 1.0}
                if verbose:
                    print("  Warning: Could not compute balance factors, using 1.0 for both")
        else:
            # Fallback: use simple intensity ratio
            if verbose:
                print("  Fallback: Using simple intensity ratio method...")
            all_cy3 = []
            all_cy5 = []
            for cyc in range(1, cyc_num + 1):
                cy3_col = f'cyc_{cyc}_cy3'
                cy5_col = f'cyc_{cyc}_cy5'
                if cy3_col in intensity_df.columns:
                    all_cy3.extend(intensity_df[cy3_col].values)
                if cy5_col in intensity_df.columns:
                    all_cy5.extend(intensity_df[cy5_col].values)
            
            if len(all_cy3) > 0 and len(all_cy5) > 0:
                cy3_median = np.median(all_cy3)
                cy5_median = np.median(all_cy5)
                
                if cy3_median > 0 and cy5_median > 0:
                    if cy3_median > cy5_median:
                        balance_cy3 = cy5_median / cy3_median
                        balance_cy5 = 1.0
                    else:
                        balance_cy3 = 1.0
                        balance_cy5 = cy3_median / cy5_median
                    channel_balance = {'cy3': balance_cy3, 'cy5': balance_cy5}
                else:
                    channel_balance = {'cy3': 1.0, 'cy5': 1.0}
            else:
                channel_balance = {'cy3': 1.0, 'cy5': 1.0}
            
            if verbose:
                print(f"  Channel balance factors: cy3={channel_balance['cy3']:.3f}, "
                      f"cy5={channel_balance['cy5']:.3f}")
    elif channel_balance is None:
        channel_balance = {'cy3': 1.0, 'cy5': 1.0}
    
    # Step 2: Extract intensity vectors and decode per round (vectorized)
    if verbose:
        print("  Decoding per-round max channel (vectorized)...")
    
    # Extract all intensity data as numpy arrays (vectorized)
    intensity_cols = [f'cyc_{cyc}_{ch}' for cyc in range(1, cyc_num + 1) for ch in channels]
    intensity_data = intensity_df[intensity_cols].values.astype(float)  # (n_points, n_features)
    
    # Apply channel balance factors (vectorized)
    balance_cy3 = channel_balance.get('cy3', 1.0)
    balance_cy5 = channel_balance.get('cy5', 1.0)
    
    # Reshape to (n_points, n_cycles, n_channels)
    n_points = len(intensity_df)
    intensity_reshaped = intensity_data.reshape(n_points, cyc_num, len(channels))
    
    # Apply balance: (n_points, n_cycles, n_channels)
    intensity_reshaped[:, :, 0] *= balance_cy3  # cy3
    intensity_reshaped[:, :, 1] *= balance_cy5  # cy5
    
    # Extract cy3 and cy5 for all points and cycles: (n_points, n_cycles)
    cy3_all = intensity_reshaped[:, :, 0]  # (n_points, n_cycles)
    cy5_all = intensity_reshaped[:, :, 1]  # (n_points, n_cycles)
    
    # Compute total intensity: (n_points, n_cycles)
    total_intensity_all = cy3_all + cy5_all
    
    # Normalize: (n_points, n_cycles)
    cy3_norm_all = np.divide(cy3_all, total_intensity_all, 
                             out=np.zeros_like(cy3_all), 
                             where=total_intensity_all > 1e-6)
    cy5_norm_all = np.divide(cy5_all, total_intensity_all,
                             out=np.zeros_like(cy5_all),
                             where=total_intensity_all > 1e-6)
    
    # Determine high/low: (n_points, n_cycles)
    threshold = 0.4
    cy3_high_all = cy3_norm_all > threshold
    cy5_high_all = cy5_norm_all > threshold
    
    # Determine bases using vectorized operations: (n_points, n_cycles)
    # A: (cy3_high & cy5_high)
    # T: (cy3_high & ~cy5_high)
    # C: (~cy3_high & cy5_high)
    # G: (~cy3_high & ~cy5_high)
    # N: (total_intensity < 1e-6)
    
    mask_low_signal = total_intensity_all < 1e-6
    mask_A = cy3_high_all & cy5_high_all & ~mask_low_signal
    mask_T = cy3_high_all & ~cy5_high_all & ~mask_low_signal
    mask_C = ~cy3_high_all & cy5_high_all & ~mask_low_signal
    mask_G = ~cy3_high_all & ~cy5_high_all & ~mask_low_signal
    
    # Create base array: (n_points, n_cycles)
    bases_array = np.full((n_points, cyc_num), 'N', dtype='U1')
    bases_array[mask_A] = 'A'
    bases_array[mask_T] = 'T'
    bases_array[mask_C] = 'C'
    bases_array[mask_G] = 'G'
    
    # Compute confidence for each cycle: (n_points, n_cycles)
    confidence_array = np.zeros((n_points, cyc_num), dtype=float)
    
    # A: confidence based on balance
    mask_A_valid = mask_A
    if np.any(mask_A_valid):
        balance_confidence = 1.0 - np.abs(cy3_norm_all[mask_A_valid] - cy5_norm_all[mask_A_valid]) / 0.5
        confidence_array[mask_A_valid] = balance_confidence
    
    # T: confidence = cy3_norm
    if np.any(mask_T):
        confidence_array[mask_T] = cy3_norm_all[mask_T]
    
    # C: confidence = cy5_norm
    if np.any(mask_C):
        confidence_array[mask_C] = cy5_norm_all[mask_C]
    
    # G: confidence based on how low both are
    if np.any(mask_G):
        g_confidence = 1.0 - np.minimum(cy3_norm_all[mask_G] + cy5_norm_all[mask_G], 0.8)
        confidence_array[mask_G] = g_confidence
    
    # Scale confidence by total intensity
    intensity_confidence_all = np.minimum(total_intensity_all / 100.0, 1.0)
    confidence_array *= intensity_confidence_all
    
    # Convert bases array to sequences
    if verbose:
        print("  Converting bases to sequences...")
    decoded_sequences = [''.join(bases_array[i, :]) for i in range(n_points)]
    
    # Compute average confidence per point
    confidences = np.mean(confidence_array, axis=1)
    
    # Step 3: Match decoded sequences to codebook (vectorized)
    if verbose:
        print("  Matching sequences to codebook (vectorized)...")
    
    # Convert to numpy arrays for vectorized operations
    confidences_array = np.array(confidences)
    
    # Filter by confidence threshold first
    valid_mask = confidences_array >= min_confidence
    
    # Initialize result arrays
    matched_genes = np.full(n_points, np.nan, dtype=object)
    matched_sequences = np.full(n_points, np.nan, dtype=object)
    final_confidences = confidences_array.copy()
    
    # Get valid sequences for matching
    valid_sequences = np.array(decoded_sequences)[valid_mask]
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_sequences) > 0:
        if verbose:
            print(f"  Matching {len(valid_sequences)} sequences (after confidence filter)...")
        
        # Step 3.1: Exact matching (fast, using set lookup)
        if verbose:
            print("  Step 3.1: Exact matching...")
        
        barcode_set = set(barcode_to_gene.keys())
        exact_matches = np.array([seq in barcode_set for seq in valid_sequences])
        exact_match_indices = np.where(exact_matches)[0]
        
        if len(exact_match_indices) > 0:
            # Fill in exact matches
            for idx in exact_match_indices:
                global_idx = valid_indices[idx]
                seq = valid_sequences[idx]
                matched_genes[global_idx] = barcode_to_gene[seq]
                matched_sequences[global_idx] = seq
        
        # Step 3.2: Hamming distance matching for non-exact matches (vectorized)
        non_exact_mask = ~exact_matches
        if np.any(non_exact_mask) and verbose:
            print(f"  Step 3.2: Hamming distance matching for {np.sum(non_exact_mask)} sequences...")
        
        # Convert barcodes to numpy array for vectorized comparison
        barcodes_list = list(barcode_to_gene.keys())
        barcodes_array = np.array([list(bc) for bc in barcodes_list])  # (n_barcodes, seq_length)
        
        # Process non-exact matches in batches to avoid memory issues
        non_exact_indices = np.where(non_exact_mask)[0]
        batch_size_hamming = 10000  # Process 10k sequences at a time for Hamming distance
        
        for batch_start in tqdm(range(0, len(non_exact_indices), batch_size_hamming),
                               desc='  Hamming matching', disable=not verbose or len(non_exact_indices) == 0):
            batch_end = min(batch_start + batch_size_hamming, len(non_exact_indices))
            batch_indices = non_exact_indices[batch_start:batch_end]
            batch_sequences = valid_sequences[batch_indices]
            
            # Convert batch sequences to array: (batch_size, seq_length)
            batch_seqs_array = np.array([list(seq) for seq in batch_sequences])
            
            # Compute Hamming distances: (batch_size, n_barcodes)
            # Using broadcasting: compare each sequence with all barcodes
            if len(batch_seqs_array) > 0 and len(barcodes_array) > 0:
                # Ensure same length
                seq_length = len(batch_seqs_array[0])
                barcodes_filtered = barcodes_array[:, :seq_length]  # Only compare up to sequence length
                
                # Compute distances: (batch_size, n_barcodes)
                # Expand dimensions for broadcasting: (batch_size, 1, seq_length) vs (1, n_barcodes, seq_length)
                batch_seqs_expanded = batch_seqs_array[:, None, :]  # (batch_size, 1, seq_length)
                barcodes_expanded = barcodes_filtered[None, :, :]    # (1, n_barcodes, seq_length)
                
                # Compute mismatches: (batch_size, n_barcodes)
                mismatches = (batch_seqs_expanded != barcodes_expanded).sum(axis=2)
                
                # Find best match for each sequence (distance <= 1)
                best_distances = np.min(mismatches, axis=1)  # (batch_size,)
                best_barcode_indices = np.argmin(mismatches, axis=1)  # (batch_size,)
                
                # Apply threshold (distance <= 1)
                valid_hamming = best_distances <= 1
                
                # Fill in results
                for i, (local_idx, global_valid_idx) in enumerate(zip(batch_indices, valid_indices[batch_indices])):
                    if valid_hamming[i]:
                        barcode_idx = best_barcode_indices[i]
                        barcode = barcodes_list[barcode_idx]
                        distance = best_distances[i]
                        
                        matched_genes[global_valid_idx] = barcode_to_gene[barcode]
                        matched_sequences[global_valid_idx] = barcode
                        # Reduce confidence for mismatches
                        final_confidences[global_valid_idx] *= (1.0 - distance * 0.2)
                    else:
                        # No match found, keep original sequence
                        matched_sequences[global_valid_idx] = batch_sequences[i]
    
    # Fill in sequences for all points (even if not matched)
    for i in range(n_points):
        if pd.isna(matched_sequences[i]):
            matched_sequences[i] = decoded_sequences[i]
    
    # Convert to lists
    matched_genes = matched_genes.tolist()
    matched_sequences = matched_sequences.tolist()
    final_confidences = final_confidences.tolist()
    
    # Step 4: Create result dataframe
    result_df = pd.DataFrame({
        'index': intensity_df['index'].values if 'index' in intensity_df.columns else intensity_df.index.values,
        'Sequence': matched_sequences,
        'Gene': matched_genes,
        'confidence': final_confidences
    })
    
    if verbose:
        mapped_count = result_df['Gene'].notna().sum()
        print(f"  Mapped {mapped_count}/{len(result_df)} points to genes "
              f"({mapped_count/len(result_df)*100:.1f}%)")
        if mapped_count > 0:
            avg_confidence = result_df[result_df['Gene'].notna()]['confidence'].mean()
            print(f"  Average confidence: {avg_confidence:.3f}")
    
    return result_df



def postcode_mapping(intensity_df, ref_file, cyc_num=10, channels=['cy3', 'cy5'],
                     num_iter=60, batch_size=15000, inference_chunk_size=100000,
                     print_training_progress=True, verbose=True, 
                     return_diagnostics=False, **kwargs):
    """
    Map genes using PoSTcode (Probabilistic Spatial Transcriptomics Decoder).
    
    This method uses a probabilistic graphical model to decode genes.
    It replicates the training logic of PoSTcode but implements a chunked inference strategy
    to handle large datasets (millions of spots) without memory errors.
    
    Parameters
    ----------
    intensity_df : pd.DataFrame
        DataFrame with intensity columns and 'index' column
    ref_file : str or Path
        Path to reference file (CSV with Barcode and Gene columns)
    cyc_num : int
        Number of cycles (default: 10)
    channels : list
        Channel names (default: ['cy3', 'cy5'])
    num_iter : int
        Number of iterations for variational inference (default: 60)
    batch_size : int
        Batch size for SVI training (default: 15000)
    inference_chunk_size : int
        Chunk size for inference step to avoid OOM (default: 100000)
    print_training_progress : bool
        If True, print training progress (default: True)
    verbose : bool
        If True, print high-level progress (default: True)
    return_diagnostics : bool
        If True, return diagnostics in the DataFrame (Entropy, etc.) (default: False)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ['index', 'Sequence', 'Gene', 'Probability', 'Probability_2', 'Entropy']
    """
    # Lazy imports — postcode + torch/pyro are heavy optional deps.
    # postcode is NOT on PyPI under this name; install the vendored HuangLab
    # fork as editable: `pip install -e <SPRINTseq>/experiments/src/postcode`.
    try:
        import torch
        import pyro
        import pyro.distributions as dist
        from pyro.infer import SVI, TraceEnum_ELBO
        from pyro.optim import Adam
        from postcode.decoding_functions import (
            model_constrained_tensor,
            auto_guide_constrained_tensor,
            train,
            e_step,
            chol_sigma_from_vec,
            kronecker_product,
            mat_sqrt,
            torch_format,
            barcodes_01_from_channels,
        )
    except ImportError as e:
        raise ImportError(
            "PoSTcode requires torch, pyro, and the postcode package. "
            "postcode is NOT on PyPI — install the vendored HuangLab fork: "
            "pip install -e <SPRINTseq>/experiments/src/postcode "
            "(or switch method to 'threshold' / 'intensity_direct' / 'per_round_max' to bypass postcode)."
        ) from e

    import numpy as np
    import itertools
    from tqdm import tqdm
    import pandas as pd

    # 2. Reshape Intensity Data -> (N, C, R)
    if verbose:
        print("  Preparing data for PoSTcode (Chunked Strategy)...")
    
    intensity_cols = [f'cyc_{cyc}_{ch}' for cyc in range(1, cyc_num + 1) for ch in channels]
    
    # Check for missing columns
    missing = [c for c in intensity_cols if c not in intensity_df.columns]
    if missing:
        raise ValueError(f"Missing intensity columns: {missing}")
        
    n_spots = len(intensity_df)
    n_channels = len(channels)
    n_rounds = cyc_num
    
    # Get raw values and reshape to (N, C, R)
    # Note: intensity_df might be large, be careful with copies
    flat_intensities = intensity_df[intensity_cols].values # (N, R*C)
    reshaped_n_r_c = flat_intensities.reshape(n_spots, n_rounds, n_channels)
    spots = reshaped_n_r_c.transpose(0, 2, 1) # (N, C, R)
    
    # 3. Prepare Barcodes -> (K, C, R) 0/1 Matrix
    ref_df = pd.read_csv(ref_file)
    
    # Identify columns
    if 'Barcode' in ref_df.columns:
        barcode_col = 'Barcode'
    elif 'barcode' in ref_df.columns:
        barcode_col = 'barcode'
    else:
        barcode_col = ref_df.columns[1]
        
    if 'Gene' in ref_df.columns:
        gene_col = 'Gene'
    elif 'gene' in ref_df.columns:
        gene_col = 'gene'
    else:
        gene_col = ref_df.columns[0]
        
    barcodes_list = ref_df[barcode_col].astype(str).str.upper().tolist()
    gene_list = ref_df[gene_col].tolist()
    
    # Mapping rule
    base_map = {'A': [1, 1], 'T': [1, 0], 'C': [0, 1], 'G': [0, 0]}
    
    K = len(barcodes_list)
    barcodes_01 = np.zeros((K, n_channels, n_rounds), dtype=int)
    
    for k, code in enumerate(barcodes_list):
        for r in range(n_rounds):
            if r < len(code):
                base = code[r]
                vals = base_map.get(base, [0, 0])
                barcodes_01[k, :, r] = vals
            else:
                barcodes_01[k, :, r] = [0, 0]

    # 4. PoSTcode Logic (Replicated from decoding_function)
    
    # Set device
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type("torch.FloatTensor")
            
    N = spots.shape[0]
    C = spots.shape[1]
    R = spots.shape[2]
    K = barcodes_01.shape[0]
    D = C * R
    
    data = torch_format(spots)
    codes = torch_format(barcodes_01)

    # Defaults from decoding_function
    up_prc_to_remove = 99.95
    modify_bkg_prior = True
    estimate_bkg = True
    estimate_additional_barcodes = None
    add_remaining_barcodes_prior = 0.05
    set_seed = 1

    # include background / any additional barcode in codebook
    if estimate_bkg:
        bkg_ind = codes.shape[0]
        codes = torch.cat((codes, torch.zeros(1, D)))
    else:
        bkg_ind = np.empty((0,), dtype=np.int32)
        
    if np.any(estimate_additional_barcodes is not None):
        inf_ind = codes.shape[0] + np.arange(estimate_additional_barcodes.shape[0])
        codes = torch.cat((codes, torch_format(estimate_additional_barcodes)))
    else:
        inf_ind = np.empty((0,), dtype=np.int32)

    # normalize spot values
    if verbose:
        print("  Normalizing data...")
        
    # We must operate on CPU for numpy percentiles usually, data is tensor
    data_cpu = data.cpu().numpy()
    
    if up_prc_to_remove < 100:
        sums = np.sum(data_cpu < np.percentile(data_cpu, up_prc_to_remove, axis=0), axis=1)
        ind_keep = np.where(sums == D)[0]
    else:
        ind_keep = np.arange(0, N)
        
    # Check if ind_keep is empty
    if len(ind_keep) == 0:
        print("  Warning: No spots kept after filtering! Using all spots.")
        ind_keep = np.arange(0, N)

    s = torch.tensor(np.percentile(data_cpu[ind_keep, :], 60, axis=0))
    max_s = torch.tensor(np.percentile(data_cpu[ind_keep, :], 99.9, axis=0))
    min_s = torch.min(data[ind_keep, :], dim=0).values
    
    # Move to device if needed
    if torch.cuda.is_available():
        s = s.cuda()
        max_s = max_s.cuda()
        # min_s is already on device because data is on device
    
    log_add = (s ** 2 - max_s * min_s) / (max_s + min_s - 2 * s)
    log_add = torch.max(-torch.min(data[ind_keep, :], dim=0).values + 1e-10, other=log_add.float())
    
    # Calculate log and norm
    # To save memory, we might process this? But normalization parameters need full stats
    # We will compute parameters on ind_keep subset
    data_log_subset = torch.log10(data[ind_keep, :] + log_add)
    data_log_mean = data_log_subset.mean(dim=0, keepdim=True)
    data_log_std = data_log_subset.std(dim=0, keepdim=True)
    
    # Free memory of subset
    del data_log_subset
    
    # 5. Model Training (SVI)
    if verbose:
        print(f"  Training model parameters (SVI) on {len(ind_keep)} spots...")
        print(f"  Iterations: {num_iter}, Batch Size: {batch_size}")

    # For training, we need normalized data for the subset
    # Recalculate normalized data for subset ONLY
    data_norm_subset = (torch.log10(data[ind_keep, :] + log_add) - data_log_mean) / data_log_std

    optim = Adam({'lr': 0.085, 'betas': [0.85, 0.99]})
    svi = SVI(model_constrained_tensor, auto_guide_constrained_tensor, optim, loss=TraceEnum_ELBO(max_plate_nesting=1))
    pyro.set_rng_seed(set_seed)
    
    losses = train(svi, num_iter, data_norm_subset, len(ind_keep), D, C, R, codes.shape[0], codes, 
                   print_training_progress, min(len(ind_keep), batch_size))
                   
    # Clean up training data
    del data_norm_subset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 6. Collect Estimated Parameters
    w_star = pyro.param('weights').detach()
    sigma_ch_v_star = pyro.param('sigma_ch_v').detach()
    sigma_ro_v_star = pyro.param('sigma_ro_v').detach()
    sigma_ro_star = chol_sigma_from_vec(sigma_ro_v_star, R)
    sigma_ch_star = chol_sigma_from_vec(sigma_ch_v_star, C)
    sigma_star = kronecker_product(sigma_ro_star, sigma_ch_star)
    codes_tr_v_star = pyro.param('codes_tr_v').detach()
    codes_tr_consts_v_star = pyro.param('codes_tr_consts_v').detach()
    theta_star = torch.matmul(codes * codes_tr_v_star + codes_tr_consts_v_star, mat_sqrt(sigma_star, D))

    # 7. Prepare Prior for Inference
    if modify_bkg_prior and w_star.shape[0] > K:
        w_star_mod = torch.cat((w_star[0:K], w_star[0:K].min().repeat(w_star.shape[0] - K)))
        w_star_mod = w_star_mod / w_star_mod.sum()
    else:
        w_star_mod = w_star

    # Handle "add_remaining_barcodes_prior" logic to determine final codes and weights
    if add_remaining_barcodes_prior > 0:
        barcodes_1234 = np.array([p for p in itertools.product(np.arange(1, C + 1), repeat=R)])
        codes_inf_np = np.array(torch_format(barcodes_01_from_channels(barcodes_1234, C, R)).cpu())
        codes_inf_np = np.concatenate((np.zeros((1, D)), codes_inf_np)) # add bkg code
        
        codes_cpu = codes.cpu()
        # Remove existing codes
        # Optimization: use set for faster lookup instead of loop
        existing_codes = set()
        for i in range(codes_cpu.shape[0]):
             existing_codes.add(tuple(codes_cpu[i].numpy().astype(int)))
             
        keep_mask = []
        for i in range(codes_inf_np.shape[0]):
            code_tuple = tuple(codes_inf_np[i].astype(int))
            keep_mask.append(code_tuple not in existing_codes)
        
        codes_inf_np = codes_inf_np[keep_mask]
        
        if not estimate_bkg:
            bkg_ind = codes_cpu.shape[0]
            inf_ind = np.append(inf_ind, codes_cpu.shape[0] + 1 + np.arange(codes_inf_np.shape[0]))
        else:
            inf_ind = np.append(inf_ind, codes_cpu.shape[0] + np.arange(codes_inf_np.shape[0]))
            
        codes_inf = torch.tensor(codes_inf_np).float()
        if torch.cuda.is_available():
             codes_inf = codes_inf.cuda()
             
        alpha = (1 - add_remaining_barcodes_prior)
        w_star_final = torch.cat((alpha * w_star_mod, torch.tensor((1 - alpha) / codes_inf.shape[0], device=w_star_mod.device).repeat(codes_inf.shape[0])))
        
        # Recalculate theta with all codes
        all_codes = torch.cat((codes, codes_inf))
        theta_final = torch.matmul(all_codes * codes_tr_v_star + codes_tr_consts_v_star.repeat(w_star_final.shape[0], 1), mat_sqrt(sigma_star, D))
        
        K_inference = w_star_final.shape[0]
    else:
        w_star_final = w_star_mod
        theta_final = theta_star
        K_inference = codes.shape[0]
        
    # 8. Chunked Inference (E-step)
    if verbose:
        print(f"  Running inference (E-step) in chunks of {inference_chunk_size}...")
        
    # Prepare result containers (on CPU)
    res_indices = np.zeros(N, dtype=int)
    res_probs_1 = np.zeros(N, dtype=float)
    res_probs_2 = np.zeros(N, dtype=float)
    res_entropy = np.zeros(N, dtype=float)
    
    # Calculate number of chunks
    n_chunks = (N + inference_chunk_size - 1) // inference_chunk_size
    
    for i in tqdm(range(n_chunks), desc="  Inference Chunks", disable=not verbose):
        start_idx = i * inference_chunk_size
        end_idx = min((i + 1) * inference_chunk_size, N)
        
        # 1. Normalize chunk
        # data[start_idx:end_idx] is already on device if set_default_tensor_type was used?
        # data was created with torch_format(spots). spots was created before set_default.
        # Check device of data
        chunk_data = data[start_idx:end_idx]
        if torch.cuda.is_available() and chunk_data.device.type == 'cpu':
            chunk_data = chunk_data.cuda()
            
        chunk_norm = (torch.log10(chunk_data + log_add) - data_log_mean) / data_log_std
        
        # 2. Run E-step
        # e_step returns (N_chunk, K_inference) probabilities
        class_probs_chunk = e_step(chunk_norm, w_star_final, theta_final, sigma_star, 
                                   chunk_norm.shape[0], K_inference, print_training_progress=False)
        
        # 3. Handle Special Classes (Logic from decoding_function)
        # We need to collapse background and infeasible classes into single columns IF we want full matrix
        # But we only want Top1/Top2/Entropy.
        # "inf_ind" and "bkg_ind" are indices in the K_inference probability vector.
        # If we just take Top1 index, we can map it later.
        
        # However, to be consistent with PoSTcode output, we should aggregate the probabilities for "infeasible"
        # The logic in decoding_function lines 226:
        # class_probs_star_s = torch.cat((..., torch.sum(class_probs_star[:, inf_ind], dim=1)...))
        # This collapses all infeasible codes into ONE probability column.
        # This is important because individual infeasible codes might have low prob, but sum might be high.
        
        # Let's replicate this collapsing on the chunk
        # K is original barcodes count (including bkg if estim_bkg=False? No)
        # K from input is barcodes_01.shape[0].
        
        # Identify ranges
        # class_probs_chunk: [0..K-1] are genes
        # Then maybe bkg_ind
        # Then inf_ind
        
        # Construct collapsed matrix: [Genes (0..K-1), Bkg, Inf, NaN]
        # But wait, logic is complex:
        # class_probs_star_s = torch.cat((
        #    class_probs_star[:, 0:K], 
        #    class_probs_star[:, bkg_ind].reshape((N, 1)), 
        #    torch.sum(class_probs_star[:, inf_ind], dim=1).reshape((N, 1))
        # ), dim=1)
        
        # We need to be careful with indices.
        # K is len(barcodes_list)
        
        # Genes
        probs_genes = class_probs_chunk[:, 0:K]
        
        # Background
        # Handle bkg_ind being int or array
        has_bkg = False
        if isinstance(bkg_ind, (int, np.integer)):
             has_bkg = True
             bkg_indices = [bkg_ind]
        elif hasattr(bkg_ind, '__len__') and len(bkg_ind) > 0:
             has_bkg = True
             bkg_indices = bkg_ind
        else:
             # handle numpy scalar or other types
             try:
                 # Check if it's a scalar by trying to convert to int
                 bkg_indices = [int(bkg_ind)]
                 has_bkg = True
             except:
                 has_bkg = False

        if has_bkg:
             probs_bkg = class_probs_chunk[:, bkg_indices].reshape(chunk_norm.shape[0], -1)
        else:
             probs_bkg = torch.zeros((chunk_norm.shape[0], 0), device=class_probs_chunk.device)
             
        # Infeasible
        # Handle inf_ind being int or array
        has_inf = False
        if isinstance(inf_ind, (int, np.integer)):
             has_inf = True
             inf_indices = [inf_ind]
        elif hasattr(inf_ind, '__len__') and len(inf_ind) > 0:
             has_inf = True
             inf_indices = inf_ind
        else:
             # handle numpy scalar
             try:
                 inf_indices = [int(inf_ind)]
                 has_inf = True
             except:
                 has_inf = False

        if has_inf:
             probs_inf = torch.sum(class_probs_chunk[:, inf_indices], dim=1, keepdim=True)
        else:
             probs_inf = torch.zeros((chunk_norm.shape[0], 0), device=class_probs_chunk.device)
             
        # Concatenate: Genes | Bkg | Inf
        # Note: If no bkg or inf, dimensions handle it.
        # The output order in PoSTcode is: Genes, Bkg, Inf
        
        collapsed_probs = torch.cat((probs_genes, probs_bkg, probs_inf), dim=1)
        
        # Handle NaN (lines 229-234)
        # nan_spot_ind = torch.unique((torch.isnan(class_probs_star_s)).nonzero(...) )
        # Ideally we don't have NaNs if data is good.
        # But let's check.
        # If we have NaNs, we add a NaN class.
        # For simplicity and performance, maybe skip NaN class creation unless needed?
        # User just wants Top1/Top2/Entropy.
        # If NaN exists, Entropy is NaN.
        
        # 4. Calculate Metrics (Top1, Top2, Entropy)
        
        # Entropy
        # Limit probabilities to avoid log(0)
        probs_safe = collapsed_probs.clamp(min=1e-10)
        entropy = -torch.sum(probs_safe * torch.log(probs_safe), dim=1)
        
        # Top 2
        # collapsed_probs shape: (N_chunk, K + has_bkg + has_inf)
        k_top = min(2, collapsed_probs.shape[1])
        top_probs, top_indices = torch.topk(collapsed_probs, k=k_top, dim=1)
        
        # Store results
        res_indices[start_idx:end_idx] = top_indices[:, 0].cpu().numpy()
        res_probs_1[start_idx:end_idx] = top_probs[:, 0].cpu().numpy()
        res_entropy[start_idx:end_idx] = entropy.cpu().numpy()
        
        if k_top >= 2:
            res_probs_2[start_idx:end_idx] = top_probs[:, 1].cpu().numpy()
        else:
            res_probs_2[start_idx:end_idx] = 0.0
            
        # Free memory
        del chunk_norm, class_probs_chunk, collapsed_probs, probs_safe, top_probs, top_indices, entropy
    
    # 9. Map Indices to Names
    # Indices 0..K-1 -> Gene Names
    # Index K -> Bkg (if exists)
    # Index K+1 -> Inf (if exists)
    
    # Determine index mapping
    # result_genes = np.full(n_spots, np.nan, dtype=object) # Gene Name
    # result_seqs = ...
    
    # Mapped arrays
    mapped_genes = np.empty(N, dtype=object)
    mapped_seqs = np.empty(N, dtype=object)
    
    # Create lookup
    # 0..K-1: genes
    gene_lookup = np.array(gene_list)
    seq_lookup = np.array(barcodes_list)
    
    # Identify special indices in the COLLAPSED matrix
    # Format: [Genes... | Bkg? | Inf? ]
    # Bkg index = K (if estimate_bkg=True)
    # Inf index = K + 1 (if estimate_bkg=True and estimate_additional=None but add_prior>0?)
    # Wait, if estimate_bkg=True, probs_bkg is added. So column K is Bkg.
    # If len(inf_ind)>0, probs_inf is added. It is next.
    
    current_idx = K
    bkg_col_idx = -1
    inf_col_idx = -1
    
    if estimate_bkg:
        bkg_col_idx = current_idx
        current_idx += 1
        
    # Check if inf_ind indicates existence of infeasible codes
    has_inf_codes = False
    if isinstance(inf_ind, (int, np.integer)):
         has_inf_codes = True
    elif hasattr(inf_ind, '__len__') and len(inf_ind) > 0:
         has_inf_codes = True
    
    if has_inf_codes:
        inf_col_idx = current_idx
        current_idx += 1
        
    # Vectorized assignment
    # 1. Genes
    gene_mask = (res_indices < K)
    mapped_genes[gene_mask] = gene_lookup[res_indices[gene_mask]]
    mapped_seqs[gene_mask] = seq_lookup[res_indices[gene_mask]]
    
    # 2. Background
    if bkg_col_idx != -1:
        bkg_mask = (res_indices == bkg_col_idx)
        mapped_genes[bkg_mask] = 'Background'
        mapped_seqs[bkg_mask] = 'Background'
        
    # 3. Infeasible
    if inf_col_idx != -1:
        inf_mask = (res_indices == inf_col_idx)
        mapped_genes[inf_mask] = 'Infeasible'
        mapped_seqs[inf_mask] = 'Infeasible'
        
    # 4. Unknown/NaN (if any other index)
    other_mask = ~(gene_mask | (res_indices == bkg_col_idx) | (res_indices == inf_col_idx))
    mapped_genes[other_mask] = np.nan
    mapped_seqs[other_mask] = np.nan
    
    # 10. Construct DataFrame
    result_df = pd.DataFrame({
        'index': intensity_df['index'].values if 'index' in intensity_df.columns else intensity_df.index.values,
        'Sequence': mapped_seqs,
        'Gene': mapped_genes,
        'Probability': res_probs_1,
        'Probability_2': res_probs_2,
        'Entropy': res_entropy
    })
    
    if verbose:
        valid_genes = result_df[~result_df['Gene'].isin(['Background', 'Infeasible']) & result_df['Gene'].notna()]
        mapped_count = len(valid_genes)
        print(f"  Mapped {mapped_count}/{len(result_df)} points to actual genes "
              f"({mapped_count/len(result_df)*100:.1f}%)")
        print(f"  Average Probability (Top 1): {result_df['Probability'].mean():.3f}")
        print(f"  Average Entropy: {result_df['Entropy'].mean():.3f}")

    if return_diagnostics:
        diagnostics = {'losses': losses}
        return result_df, diagnostics
        
    return result_df


def probabilistic_mapping(intensity_df, ref_file, temperature=1.0, min_prob=0.5,
                         cyc_num=10, channels=['cy3', 'cy5'], match_threshold=0.8,
                         verbose=True):
    """
    Map genes using probabilistic base calling.
    
    This method converts intensity values to probabilities and uses probabilistic
    matching similar to Illumina's Bustard algorithm.
    
    Parameters
    ----------
    intensity_df : pd.DataFrame
        DataFrame with intensity columns and 'index' column
    ref_file : str or Path
        Path to reference file
    temperature : float
        Temperature parameter for probability distribution (default: 1.0)
    min_prob : float
        Minimum probability threshold (default: 0.5)
    cyc_num : int
        Number of cycles (default: 10)
    channels : list
        Channel names (default: ['cy3', 'cy5'])
    match_threshold : float
        Probability threshold for sequence matching (default: 0.8)
    verbose : bool
        Print progress (default: True)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ['index', 'Sequence', 'Gene']
    """
    # TODO: Implement probabilistic_mapping
    raise NotImplementedError("probabilistic_mapping is not yet implemented")


def map_genes(intensity_df, ref_file, method='threshold', **kwargs):
    """
    Unified interface for gene mapping methods.
    
    Parameters
    ----------
    intensity_df : pd.DataFrame
        DataFrame with intensity columns and 'index' column
    ref_file : str or Path
        Path to reference file
    method : str
        Mapping method:
        - 'threshold': Fixed or adaptive threshold-based base calling + Hamming distance matching
        - 'intensity_direct': Direct intensity similarity matching (starfish MetricDistance-style)
        - 'per_round_max': Per-round max channel decoding (starfish PerRoundMaxChannel-style)
        - 'probabilistic': Probabilistic base calling (to be implemented)
    **kwargs : dict
        Method-specific parameters
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ['index', 'Sequence', 'Gene']
        Additional columns may include 'similarity' or 'confidence' depending on method
    """
    if method == 'threshold':
        return threshold_mapping(intensity_df, ref_file, **kwargs)
    elif method == 'intensity_direct':
        return intensity_direct_mapping(intensity_df, ref_file, **kwargs)
    elif method == 'per_round_max':
        return per_round_max_channel_mapping(intensity_df, ref_file, **kwargs)
    elif method == 'probabilistic':
        return probabilistic_mapping(intensity_df, ref_file, **kwargs)
    elif method == 'postcode':
        return postcode_mapping(intensity_df, ref_file, **kwargs)
    else:
        raise ValueError(f"Unknown mapping method: {method}. "
                       f"Available methods: 'threshold', 'intensity_direct', "
                       f"'per_round_max', 'probabilistic', 'postcode'")


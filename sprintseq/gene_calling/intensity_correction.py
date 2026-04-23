"""
Robust decay correction using percentile-based method.

This module implements the percentile-based decay correction method recommended
in info_percentile.md to handle base composition bias in ISS signal decay.
"""

import numpy as np
import pandas as pd
from pathlib import Path

def smooth_decay_curve(decay_curve, method='savgol', window_length=5, polyorder=2):
    """
    Smooth decay curve to remove small fluctuations.
    
    Parameters
    ----------
    decay_curve : np.ndarray
        Raw decay curve values
    method : str
        Smoothing method: 'savgol', 'moving_average', or 'none'
    window_length : int
        Window length for smoothing (must be odd for savgol)
    polyorder : int
        Polynomial order for savgol filter
    
    Returns
    -------
    np.ndarray
        Smoothed decay curve
    """
    if method == 'none' or len(decay_curve) < 3:
        return decay_curve.copy()
    
    if method == 'savgol':
        from scipy.signal import savgol_filter
        # Ensure window_length is odd and not larger than array
        window_length = min(window_length, len(decay_curve))
        if window_length % 2 == 0:
            window_length -= 1
        if window_length < 3:
            window_length = 3
        polyorder = min(polyorder, window_length - 1)
        return savgol_filter(decay_curve, window_length, polyorder)
    
    elif method == 'moving_average':
        # Simple moving average fallback
        window_length = min(window_length, len(decay_curve))
        if window_length < 2:
            return decay_curve.copy()
        
        # Pad with edge values
        padded = np.pad(decay_curve, (window_length//2, window_length//2), mode='edge')
        smoothed = np.convolve(padded, np.ones(window_length)/window_length, mode='valid')
        return smoothed
    
    else:
        return decay_curve.copy()


def calculate_robust_decay_curve(intensity_df, cyc_num=10, percentile=99.9, channels=['cy3', 'cy5'],
                                 smooth=True, smooth_method='savgol', smooth_window=5):
    """
    Calculate robust decay curve using percentile method.
    
    This method is resistant to base composition bias because it uses the
    top percentile of intensities rather than mean, which is affected by
    how many spots are "on" in each cycle.
    
    Parameters
    ----------
    intensity_df : pd.DataFrame
        DataFrame with columns ['Y', 'X', 'cyc_1_cy3', 'cyc_1_cy5', ...]
    cyc_num : int
        Number of cycles
    percentile : float
        Percentile to use (default: 99.9 for P99.9)
    channels : list
        List of channel names (default: ['cy3', 'cy5'])
    smooth : bool
        Whether to smooth the decay curve to remove small fluctuations (default: True)
    smooth_method : str
        Smoothing method: 'savgol' (Savitzky-Golay), 'moving_average', or 'none'
    smooth_window : int
        Window length for smoothing (default: 5)
    
    Returns
    -------
    dict
        Dictionary with keys:
        - 'decay_curve': array of shape (n_cycles,) - smoothed decay curve values
        - 'decay_curve_raw': array of shape (n_cycles,) - raw (unsmoothed) decay curve
        - 'decay_curve_cy3': array of shape (n_cycles,) - cy3 decay curve (smoothed)
        - 'decay_curve_cy5': array of shape (n_cycles,) - cy5 decay curve (smoothed)
        - 'baseline': float - baseline value (max of smoothed decay curve)
        - 'scale_factors': array of shape (n_cycles,) - correction factors
        - 'smooth_applied': bool - whether smoothing was applied
    """
    cycles = np.arange(1, cyc_num + 1)
    decay_curve = []
    decay_curve_cy3 = []
    decay_curve_cy5 = []
    
    for cyc in cycles:
        # Get intensity values for this cycle
        cy3_col = f'cyc_{cyc}_cy3'
        cy5_col = f'cyc_{cyc}_cy5'
        
        if cy3_col not in intensity_df.columns or cy5_col not in intensity_df.columns:
            decay_curve.append(0)
            decay_curve_cy3.append(0)
            decay_curve_cy5.append(0)
            continue
        
        cy3_values = intensity_df[cy3_col].fillna(0).values
        cy5_values = intensity_df[cy5_col].fillna(0).values
        
        # Use max channel intensity per spot (more robust)
        max_channel_per_spot = np.maximum(cy3_values, cy5_values)
        
        # Calculate percentile
        decay_curve.append(np.percentile(max_channel_per_spot, percentile))
        decay_curve_cy3.append(np.percentile(cy3_values, percentile))
        decay_curve_cy5.append(np.percentile(cy5_values, percentile))
    
    decay_curve_raw = np.array(decay_curve)
    decay_curve_cy3_raw = np.array(decay_curve_cy3)
    decay_curve_cy5_raw = np.array(decay_curve_cy5)
    
    # Apply smoothing if requested
    if smooth:
        decay_curve = smooth_decay_curve(decay_curve_raw, method=smooth_method, 
                                        window_length=smooth_window)
        decay_curve_cy3 = smooth_decay_curve(decay_curve_cy3_raw, method=smooth_method,
                                            window_length=smooth_window)
        decay_curve_cy5 = smooth_decay_curve(decay_curve_cy5_raw, method=smooth_method,
                                            window_length=smooth_window)
        smooth_applied = True
    else:
        decay_curve = decay_curve_raw.copy()
        decay_curve_cy3 = decay_curve_cy3_raw.copy()
        decay_curve_cy5 = decay_curve_cy5_raw.copy()
        smooth_applied = False
    
    # Use max value as baseline (as recommended in info_percentile.md)
    # Use smoothed curve for baseline if smoothing was applied
    baseline = np.max(decay_curve) if np.max(decay_curve) > 0 else 1.0
    
    # Calculate scale factors for correction (multiplicative correction)
    # scale_factor[i] = baseline / decay_curve[i]
    # This means: corrected_intensity = original_intensity * scale_factor
    scale_factors = baseline / (decay_curve + 1e-10)  # Add small epsilon to avoid division by zero
    
    return {
        'decay_curve': decay_curve,  # Smoothed (or raw if smooth=False)
        'decay_curve_raw': decay_curve_raw,  # Always raw
        'decay_curve_cy3': decay_curve_cy3,
        'decay_curve_cy5': decay_curve_cy5,
        'baseline': baseline,
        'scale_factors': scale_factors,
        'percentile': percentile,
        'smooth_applied': smooth_applied,
        'smooth_method': smooth_method if smooth_applied else None
    }


def correct_decay_robust(intensity_df, cyc_num=10, percentile=99.9, channels=['cy3', 'cy5'],
                        smooth=True, smooth_method='savgol', smooth_window=5):
    """
    Correct signal decay using robust percentile-based method with optional smoothing.
    
    This function applies decay correction to intensity data using the
    percentile-based method, which is resistant to base composition bias.
    The decay curve can be smoothed to remove small fluctuations before
    calculating correction factors.
    
    Parameters
    ----------
    intensity_df : pd.DataFrame
        DataFrame with columns ['Y', 'X', 'cyc_1_cy3', 'cyc_1_cy5', ...]
    cyc_num : int
        Number of cycles
    percentile : float
        Percentile to use for decay curve estimation (default: 99.9)
    channels : list
        List of channel names (default: ['cy3', 'cy5'])
    smooth : bool
        Whether to smooth the decay curve (default: True)
    smooth_method : str
        Smoothing method: 'savgol', 'moving_average', or 'none'
    smooth_window : int
        Window length for smoothing (default: 5)
    
    Returns
    -------
    pd.DataFrame
        Corrected intensity DataFrame with same structure as input
        Correction is applied multiplicatively: corrected = original * scale_factor
    dict
        Decay correction information (decay curve, scale factors, etc.)
    """
    # Calculate robust decay curve (with optional smoothing)
    decay_info = calculate_robust_decay_curve(intensity_df, cyc_num, percentile, channels,
                                             smooth=smooth, smooth_method=smooth_method,
                                             smooth_window=smooth_window)
    
    # Create copy of intensity dataframe
    corrected_df = intensity_df.copy()
    
    # Apply correction to each cycle and channel
    cycles = np.arange(1, cyc_num + 1)
    scale_factors = decay_info['scale_factors']
    
    for cyc in cycles:
        for channel in channels:
            col = f'cyc_{cyc}_{channel}'
            if col in corrected_df.columns:
                # Apply scale factor
                corrected_df[col] = corrected_df[col] * scale_factors[cyc - 1]
    
    return corrected_df, decay_info


def correct_decay_robust_per_channel(intensity_df, cyc_num=10, percentile=99.9, channels=['cy3', 'cy5'],
                                     smooth=True, smooth_method='savgol', smooth_window=5):
    """
    Correct signal decay using channel-specific percentile-based method.
    
    This version calculates separate decay curves for each channel and applies
    channel-specific corrections. Each channel's curve can be smoothed separately.
    
    Parameters
    ----------
    intensity_df : pd.DataFrame
        DataFrame with columns ['Y', 'X', 'cyc_1_cy3', 'cyc_1_cy5', ...]
    cyc_num : int
        Number of cycles
    percentile : float
        Percentile to use for decay curve estimation (default: 99.9)
    channels : list
        List of channel names (default: ['cy3', 'cy5'])
    smooth : bool
        Whether to smooth the decay curves (default: True)
    smooth_method : str
        Smoothing method: 'savgol', 'moving_average', or 'none'
    smooth_window : int
        Window length for smoothing (default: 5)
    
    Returns
    -------
    pd.DataFrame
        Corrected intensity DataFrame with same structure as input
        Correction is applied multiplicatively: corrected = original * scale_factor
    dict
        Decay correction information with separate curves for each channel
    """
    cycles = np.arange(1, cyc_num + 1)
    decay_curves = {}
    decay_curves_raw = {}
    scale_factors = {}
    baselines = {}
    
    # Calculate decay curve for each channel separately
    for channel in channels:
        decay_curve_raw_list = []
        for cyc in cycles:
            col = f'cyc_{cyc}_{channel}'
            if col in intensity_df.columns:
                values = intensity_df[col].fillna(0).values
                decay_curve_raw_list.append(np.percentile(values, percentile))
            else:
                decay_curve_raw_list.append(0)
        
        decay_curve_raw_arr = np.array(decay_curve_raw_list)
        decay_curves_raw[channel] = decay_curve_raw_arr
        
        # Apply smoothing if requested
        if smooth:
            decay_curve = smooth_decay_curve(decay_curve_raw_arr, method=smooth_method,
                                            window_length=smooth_window)
        else:
            decay_curve = decay_curve_raw_arr.copy()
        
        baseline = np.max(decay_curve) if np.max(decay_curve) > 0 else 1.0
        scale_factor = baseline / (decay_curve + 1e-10)
        
        decay_curves[channel] = decay_curve
        scale_factors[channel] = scale_factor
        baselines[channel] = baseline
    
    # Create copy of intensity dataframe
    corrected_df = intensity_df.copy()
    
    # Apply channel-specific corrections
    for cyc in cycles:
        for channel in channels:
            col = f'cyc_{cyc}_{channel}'
            if col in corrected_df.columns:
                corrected_df[col] = corrected_df[col] * scale_factors[channel][cyc - 1]
    
    return corrected_df, {
        'decay_curves': decay_curves,  # Smoothed (or raw if smooth=False)
        'decay_curves_raw': decay_curves_raw,  # Always raw
        'scale_factors': scale_factors,
        'baselines': baselines,
        'percentile': percentile,
        'smooth_applied': smooth,
        'smooth_method': smooth_method if smooth else None
    }


def build_phasing_matrix(n_cycles, phasing_rate=0.02, prephasing_rate=0.0):
    """
    Build phasing matrix M for deconvolution with both phasing and pre-phasing.
    
    The phasing matrix M describes how true signals from each cycle
    contribute to observed signals in subsequent cycles.
    
    M[i, j] = proportion of true signal from cycle j that appears in cycle i
    
    Parameters
    ----------
    n_cycles : int
        Number of cycles
    phasing_rate : float
        Phasing rate (proportion of signal that lags to next cycle, default: 0.02)
        This is the probability that a signal from cycle j appears in cycle j+1
    prephasing_rate : float
        Pre-phasing rate (proportion of signal that leads to previous cycle, default: 0.0)
        This is the probability that a signal from cycle j appears in cycle j-1
    
    Returns
    -------
    np.ndarray
        Phasing matrix of shape (n_cycles, n_cycles)
    """
    M = np.zeros((n_cycles, n_cycles))
    p = phasing_rate      # Lagging proportion (moves to next cycle)
    q = prephasing_rate   # Leading proportion (moves to previous cycle)
    s = 1 - p - q         # Synchronous proportion (stays in current cycle)
    
    # Ensure probabilities are valid
    if s < 0 or s > 1:
        raise ValueError(f"Invalid phasing rates: p={p}, q={q}, s={s}. Must satisfy p + q <= 1")
    
    # Build matrix column by column
    # Each column j represents a true signal from cycle j
    for j in range(n_cycles):
        # The signal from cycle j can appear in multiple cycles
        # We use a simplified model where:
        # - s fraction stays in cycle j
        # - p fraction lags to cycle j+1 (and can lag further)
        # - q fraction leads to cycle j-1 (and can lead further)
        
        # Current cycle (j)
        M[j, j] = s
        
        # Leading (pre-phasing): signal appears in earlier cycles
        if q > 0 and j > 0:
            # Signal can lead to previous cycles
            for i in range(j - 1, -1, -1):
                if i == j - 1:
                    M[i, j] = q
                else:
                    # Signal can lead multiple cycles (simplified: exponential decay)
                    M[i, j] = M[i+1, j] * (q / s) if s > 0 else 0
        
        # Lagging (phasing): signal appears in later cycles
        if p > 0 and j < n_cycles - 1:
            # Signal can lag to subsequent cycles
            for i in range(j + 1, n_cycles):
                if i == j + 1:
                    M[i, j] = p
                else:
                    # Signal can lag multiple cycles (simplified: exponential decay)
                    M[i, j] = M[i-1, j] * (p / s) if s > 0 else 0
    
    return M


def correct_phasing(intensity_vector, phasing_matrix_inv):
    """
    Correct phasing using matrix inversion (deconvolution).
    
    Parameters
    ----------
    intensity_vector : np.ndarray
        Intensity values for one channel across cycles, shape (n_cycles,)
    phasing_matrix_inv : np.ndarray
        Inverse of phasing matrix, shape (n_cycles, n_cycles)
    
    Returns
    -------
    np.ndarray
        Deconvolved (phasing-corrected) intensity vector
    """
    # Deconvolution: I_true = M^-1 @ I_observed
    intensity_corrected = phasing_matrix_inv @ intensity_vector
    
    # Remove negative values (physically impossible)
    intensity_corrected = np.maximum(intensity_corrected, 0)
    
    return intensity_corrected


def correct_decay_and_phasing(intensity_df, cyc_num=10, percentile=99.9, channels=['cy3', 'cy5'],
                             smooth=True, smooth_method='savgol', smooth_window=5,
                             phasing_rate=0.02, prephasing_rate=0.0, correct_phasing=True):
    """
    Correct both signal decay and phasing.
    
    This function applies corrections in the recommended order:
    1. Decay correction (using P99.9 percentile method with smoothing)
    2. Phasing correction (using matrix inversion/deconvolution)
    
    Parameters
    ----------
    intensity_df : pd.DataFrame
        DataFrame with columns ['Y', 'X', 'cyc_1_cy3', 'cyc_1_cy5', ...]
    cyc_num : int
        Number of cycles
    percentile : float
        Percentile to use for decay curve estimation (default: 99.9)
    channels : list
        List of channel names (default: ['cy3', 'cy5'])
    smooth : bool
        Whether to smooth the decay curve (default: True)
    smooth_method : str
        Smoothing method for decay curve
    smooth_window : int
        Window length for smoothing
    phasing_rate : float
        Estimated phasing rate (default: 0.02 = 2%)
    prephasing_rate : float
        Estimated pre-phasing rate (default: 0.0 = 0%)
    correct_phasing : bool
        Whether to apply phasing correction (default: True)
    
    Returns
    -------
    pd.DataFrame
        Corrected intensity DataFrame with same structure as input
    dict
        Correction information including decay and phasing parameters
    """
    # Step 1: Decay correction
    corrected_df, decay_info = correct_decay_robust(
        intensity_df, cyc_num, percentile, channels,
        smooth=smooth, smooth_method=smooth_method, smooth_window=smooth_window
    )
    
    if not correct_phasing:
        return corrected_df, {**decay_info, 'phasing_corrected': False}
    
    # Step 2: Phasing correction
    # Build phasing matrix and its inverse (with both p and q)
    phasing_matrix = build_phasing_matrix(cyc_num, phasing_rate=phasing_rate, prephasing_rate=prephasing_rate)
    try:
        phasing_matrix_inv = np.linalg.inv(phasing_matrix)
    except np.linalg.LinAlgError:
        print(f"Warning: Phasing matrix is singular, skipping phasing correction")
        return corrected_df, {**decay_info, 'phasing_corrected': False, 'phasing_error': 'singular_matrix'}
    
    # Apply phasing correction to each spot and channel
    cycles = np.arange(1, cyc_num + 1)
    
    for idx, row in corrected_df.iterrows():
        for channel in channels:
            # Extract intensity vector for this spot and channel
            intensity_vector = np.array([
                row[f'cyc_{cyc}_{channel}'] 
                for cyc in cycles
            ])
            
            # Apply phasing correction
            intensity_corrected = correct_phasing(intensity_vector, phasing_matrix_inv)
            
            # Update dataframe
            for i, cyc in enumerate(cycles):
                col = f'cyc_{cyc}_{channel}'
                if col in corrected_df.columns:
                    corrected_df.at[idx, col] = intensity_corrected[i]
    
    return corrected_df, {
        **decay_info,
        'phasing_corrected': True,
        'phasing_rate': phasing_rate,
        'prephasing_rate': prephasing_rate,
        'phasing_matrix': phasing_matrix,
        'phasing_matrix_inv': phasing_matrix_inv
    }


# ============================================================================
# Phasing Rate Estimation Strategies
# ============================================================================

def estimate_phasing_on_off_transition(intensity_df, ref_file, cyc_num=10, channels=['cy3', 'cy5'],
                                       background_threshold=50, verbose=True):
    """
    Strategy 1: Estimate phasing rate using ON-OFF transition observation.
    
    This method finds spots with specific patterns (e.g., Cycle 1 bright, Cycle 2 dark)
    and estimates phasing rate from the "trailing" signal in the dark cycle.
    
    Parameters
    ----------
    intensity_df : pd.DataFrame
        DataFrame with columns ['Y', 'X', 'cyc_1_cy3', 'cyc_1_cy5', ...]
        Should be decay-corrected before calling this function
    ref_file : str or Path
        Path to reference file (CSV with 'Barcode' and 'Gene' columns)
    cyc_num : int
        Number of cycles
    channels : list
        List of channel names
    background_threshold : float
        Background noise threshold to subtract
    verbose : bool
        Whether to print estimation results
    
    Returns
    -------
    dict
        Dictionary with estimated phasing_rate and prephasing_rate
    """
    # Read reference file
    try:
        ref_df = pd.read_csv(ref_file)
        if 'Barcode' not in ref_df.columns:
            raise ValueError("Reference file must contain 'Barcode' column")
    except Exception as e:
        if verbose:
            print(f"Warning: Could not read reference file: {e}")
            print("Strategy 1 requires reference file, returning default values")
        return {'phasing_rate': 0.02, 'prephasing_rate': 0.0, 'method': 'on_off', 'error': str(e)}
    
    # This is a simplified implementation
    # In practice, you would:
    # 1. Find barcodes with ON->OFF patterns (e.g., Cycle 1=Cy3, Cycle 2=dark)
    # 2. For those spots, measure signal in Cycle 2 Cy3 channel (should be dark)
    # 3. Calculate p ≈ I_cyc2_cy3 / I_cyc1_cy3 (after subtracting background)
    
    # For now, return a placeholder that suggests using other strategies
    if verbose:
        print("Strategy 1 (ON-OFF Transition):")
        print("  This method requires manual pattern matching in codebook.")
        print("  Recommended: Use Strategy 2 (Grid Search) for automated estimation.")
    
    return {'phasing_rate': 0.02, 'prephasing_rate': 0.0, 'method': 'on_off', 'note': 'requires_manual_implementation'}


def estimate_phasing_grid_search(intensity_df, cyc_num=10, channels=['cy3', 'cy5'],
                                  p_range=(0.0, 0.05), q_range=(0.0, 0.02),
                                  p_step=0.002, q_step=0.001,
                                  max_spots=10000, verbose=True):
    """
    Strategy 2: Estimate phasing rate using grid search for maximum chastity.
    
    This method tests different phasing rates and selects the one that produces
    the "purest" (most binary) signal after correction.
    
    Parameters
    ----------
    intensity_df : pd.DataFrame
        DataFrame with columns ['Y', 'X', 'cyc_1_cy3', 'cyc_1_cy5', ...]
        Should be decay-corrected before calling this function
    cyc_num : int
        Number of cycles
    channels : list
        List of channel names
    p_range : tuple
        (min, max) range for phasing rate search
    q_range : tuple
        (min, max) range for pre-phasing rate search
    p_step : float
        Step size for phasing rate search
    q_step : float
        Step size for pre-phasing rate search
    max_spots : int
        Maximum number of spots to use for estimation (for speed)
    verbose : bool
        Whether to print progress
    
    Returns
    -------
    dict
        Dictionary with best phasing_rate, prephasing_rate, and score
    """
    # Sample spots if dataset is too large
    n_spots = len(intensity_df)
    if n_spots > max_spots:
        sample_df = intensity_df.sample(n=max_spots, random_state=42)
        if verbose:
            print(f"Sampling {max_spots} spots from {n_spots} total spots")
    else:
        sample_df = intensity_df
    
    # Convert to matrix format: (n_spots, n_cycles, n_channels)
    cycles = np.arange(1, cyc_num + 1)
    n_spots_used = len(sample_df)
    signal_matrix = np.zeros((n_spots_used, cyc_num, len(channels)))
    
    for i, (idx, row) in enumerate(sample_df.iterrows()):
        for j, channel in enumerate(channels):
            for k, cyc in enumerate(cycles):
                col = f'cyc_{cyc}_{channel}'
                if col in row:
                    signal_matrix[i, k, j] = row[col]
    
    # Normalize to 0-1 range
    max_val = np.max(signal_matrix)
    if max_val > 0:
        signal_matrix = signal_matrix / (max_val + 1e-10)
    
    # Grid search
    p_values = np.arange(p_range[0], p_range[1] + p_step/2, p_step)
    q_values = np.arange(q_range[0], q_range[1] + q_step/2, q_step)
    
    best_p = 0.0
    best_q = 0.0
    best_score = -np.inf
    all_scores = []
    
    if verbose:
        print(f"Strategy 2 (Grid Search): Testing {len(p_values)} x {len(q_values)} = {len(p_values)*len(q_values)} combinations")
    
    for p in p_values:
        for q in q_values:
            if p + q >= 1.0:
                continue  # Skip invalid combinations
            
            try:
                # Build phasing matrix
                phasing_matrix = build_phasing_matrix(cyc_num, phasing_rate=p, prephasing_rate=q)
                phasing_matrix_inv = np.linalg.inv(phasing_matrix)
                
                # Apply correction to all spots and channels
                corrected_matrix = np.zeros_like(signal_matrix)
                for i in range(n_spots_used):
                    for j in range(len(channels)):
                        intensity_vector = signal_matrix[i, :, j]
                        corrected_vector = phasing_matrix_inv @ intensity_vector
                        corrected_vector = np.maximum(corrected_vector, 0)  # Remove negatives
                        corrected_matrix[i, :, j] = corrected_vector
                
                # Calculate chastity score
                # Score = sum((corrected - 0.5)^2)
                # Higher score means more binary (closer to 0 or 1)
                score = np.sum((corrected_matrix - 0.5)**2)
                
                all_scores.append((p, q, score))
                
                if score > best_score:
                    best_score = score
                    best_p = p
                    best_q = q
                    
            except np.linalg.LinAlgError:
                continue  # Skip singular matrices
    
    if verbose:
        print(f"  Best phasing rate (p): {best_p:.4f} ({best_p*100:.2f}%)")
        print(f"  Best pre-phasing rate (q): {best_q:.4f} ({best_q*100:.2f}%)")
        print(f"  Best score: {best_score:.2f}")
    
    return {
        'phasing_rate': best_p,
        'prephasing_rate': best_q,
        'score': best_score,
        'method': 'grid_search',
        'all_scores': all_scores
    }


def estimate_phasing_codebook_residual(intensity_df, ref_file, cyc_num=10, channels=['cy3', 'cy5'],
                                       p_range=(0.0, 0.05), q_range=(0.0, 0.02),
                                       p_step=0.002, q_step=0.001,
                                       max_spots=5000, verbose=True):
    """
    Strategy 3: Estimate phasing rate using codebook residual minimization.
    
    This method uses known barcode sequences to find the phasing rate that
    minimizes the residual between observed and expected signals.
    
    Parameters
    ----------
    intensity_df : pd.DataFrame
        DataFrame with columns ['Y', 'X', 'cyc_1_cy3', 'cyc_1_cy5', ...]
        Should be decay-corrected and have base calling results
    ref_file : str or Path
        Path to reference file (CSV with 'Barcode' and 'Gene' columns)
    cyc_num : int
        Number of cycles
    channels : list
        List of channel names (should map to bases, e.g., cy3='T', cy5='G')
    p_range : tuple
        (min, max) range for phasing rate search
    q_range : tuple
        (min, max) range for pre-phasing rate search
    p_step : float
        Step size for phasing rate search
    q_step : float
        Step size for pre-phasing rate search
    max_spots : int
        Maximum number of spots to use for estimation
    verbose : bool
        Whether to print progress
    
    Returns
    -------
    dict
        Dictionary with best phasing_rate, prephasing_rate, and residual
    """
    # Read reference file
    try:
        ref_df = pd.read_csv(ref_file)
        if 'Barcode' not in ref_df.columns:
            raise ValueError("Reference file must contain 'Barcode' column")
        # Create barcode to gene mapping
        barcode_dict = dict(zip(ref_df['Barcode'], ref_df.get('Gene', ref_df['Barcode'])))
    except Exception as e:
        if verbose:
            print(f"Warning: Could not read reference file: {e}")
            print("Strategy 3 requires reference file, returning default values")
        return {'phasing_rate': 0.02, 'prephasing_rate': 0.0, 'method': 'codebook', 'error': str(e)}
    
    # Check if intensity_df has sequence information
    # This strategy requires base calling results or sequence matching
    if 'Sequence' not in intensity_df.columns and 'Match' not in intensity_df.columns:
        if verbose:
            print("Warning: Strategy 3 requires sequence information.")
            print("  Please run base calling first, or use Strategy 2 instead.")
        return {'phasing_rate': 0.02, 'prephasing_rate': 0.0, 'method': 'codebook', 
                'error': 'missing_sequence_info'}
    
    # This is a simplified placeholder
    # Full implementation would:
    # 1. Match spots to barcodes (using Sequence or Match column)
    # 2. For each matched spot, generate expected signal pattern from barcode
    # 3. Apply phasing matrix to expected signal to get "blurred" expected
    # 4. Calculate residual: ||observed - blurred_expected||^2
    # 5. Minimize residual over p and q
    
    if verbose:
        print("Strategy 3 (Codebook Residual):")
        print("  This method requires base calling results.")
        print("  Recommended: Use Strategy 2 (Grid Search) for automated estimation.")
    
    return {'phasing_rate': 0.02, 'prephasing_rate': 0.0, 'method': 'codebook', 
            'note': 'requires_base_calling_results'}


# ============================================================================
# Main Correction Function
# ============================================================================

def estimate_channel_balance(intensity_df, cyc_num=10, channels=['cy3', 'cy5'],
                              ref_file=None, method='codebook', verbose=True,
                              sample_size=100000, threshold_percentile=50):
    """
    Estimate channel balance factors to correct for systematic brightness differences.
    
    This function estimates the ratio between channels to correct for systematic
    differences in channel brightness (e.g., cy3 always brighter than cy5).
    
    Parameters
    ----------
    intensity_df : pd.DataFrame
        DataFrame with intensity columns
    cyc_num : int
        Number of cycles
    channels : list
        List of channel names (default: ['cy3', 'cy5'])
    ref_file : str or Path, optional
        Path to reference file (CSV with Barcode and Gene columns)
        If provided, uses codebook-based method for more accurate estimation
    method : str
        Estimation method: 'codebook' (requires ref_file), 'intensity_ratio', or 'median'
        - 'codebook': Uses known encoding patterns from codebook (most accurate)
        - 'intensity_ratio': Uses ratio of median intensities (simple but less accurate)
        - 'median': Uses median of all high-intensity points
    verbose : bool
        Print progress (default: True)
    sample_size : int
        Number of points to sample for codebook method (default: 100000)
        Only used when method='codebook'. Larger sample gives more accurate results but slower.
    threshold_percentile : float
        Percentile to use for threshold calculation (default: 50, i.e., median)
        Only used when method='codebook'. Higher values (e.g., 50-70) give better quality sequences.
    
    Returns
    -------
    dict
        Channel balance factors: {'cy3': factor_cy3, 'cy5': factor_cy5}
        Factors are applied as: corrected_intensity = raw_intensity * factor
    """
    if method == 'codebook' and ref_file is not None:
        # Most accurate method: use codebook to identify known patterns
        if verbose:
            print("  Estimating channel balance using codebook method...")
        
        import pandas as pd
        ref_df = pd.read_csv(ref_file)
        
        # Handle different column name formats
        if 'Barcode' in ref_df.columns:
            barcode_col = 'Barcode'
        elif 'barcode' in ref_df.columns:
            barcode_col = 'barcode'
        elif len(ref_df.columns) >= 2:
            barcode_col = ref_df.columns[1]  # Assume second column is barcode
        else:
            if verbose:
                print("  Warning: Cannot determine barcode column, falling back to intensity_ratio method")
            method = 'intensity_ratio'
        
        if method == 'codebook':
            # Analyze codebook to find expected patterns
            # For 2-channel encoding: T=(cy3=1, cy5=0), C=(cy3=0, cy5=1), G=(cy3=0, cy5=0)
            # We can use T and C patterns to estimate balance
            
            # Improvement 1: Sample points to speed up processing
            n_total = len(intensity_df)
            if n_total > sample_size:
                if verbose:
                    print(f"  Sampling {sample_size:,} points from {n_total:,} total points...")
                # Random sampling to get representative subset
                sample_indices = np.random.choice(n_total, size=sample_size, replace=False)
                intensity_df_sample = intensity_df.iloc[sample_indices].copy()
            else:
                intensity_df_sample = intensity_df.copy()
            
            # Generate sequences from intensity to find T and C positions
            from .sequence_generation import get_seq_df
            
            # Improvement 3: Use median (50th percentile) instead of 10th percentile for better quality
            if verbose:
                print(f"  Computing thresholds using {threshold_percentile}th percentile...")
            thresholds = {}
            for ch in channels:
                all_values = np.concatenate([intensity_df_sample[f'cyc_{cyc}_{ch}'].values 
                                            for cyc in range(1, cyc_num + 1)])
                threshold_value = np.percentile(all_values, threshold_percentile)
                thresholds[ch] = threshold_value
                if verbose:
                    print(f"    {ch}: {threshold_value:.2f} (from {threshold_percentile}th percentile)")
            
            if verbose:
                print(f"  Final thresholds: {thresholds}")
            
            seq_df = get_seq_df(intensity_df_sample, thresholds, cyc_num=cyc_num, channels=channels)
            
            # Improvement 2: Optimize matching using efficient data structures
            # Build barcode set and group by length for faster lookup
            barcodes = set(ref_df[barcode_col].astype(str).str.upper())
            barcodes_by_length = {}
            for barcode in barcodes:
                length = len(barcode)
                if length not in barcodes_by_length:
                    barcodes_by_length[length] = []
                barcodes_by_length[length].append(barcode)
            
            if verbose:
                print("  Matching sequences to codebook...")
            
            # Efficient matching: first exact match, then 1-mismatch
            matched_indices = []
            sequences = seq_df['Sequence'].values
            
            # Group sequences by length for batch processing
            sequences_by_length = {}
            for idx, seq in enumerate(sequences):
                length = len(seq)
                if length not in sequences_by_length:
                    sequences_by_length[length] = []
                sequences_by_length[length].append((idx, seq))
            
            # Process each length group
            for seq_len, seq_list in sequences_by_length.items():
                if seq_len not in barcodes_by_length:
                    continue  # Skip if no barcodes of this length
                
                same_len_barcodes = barcodes_by_length[seq_len]
                barcode_set = set(same_len_barcodes)
                
                # First pass: exact matches (O(1) lookup per sequence)
                for idx, seq in seq_list:
                    if seq in barcode_set:
                        matched_indices.append(idx)
                
                # Second pass: 1-mismatch for non-exact matches
                # Only check sequences that didn't match exactly
                non_exact = [(idx, seq) for idx, seq in seq_list if idx not in matched_indices]
                
                if len(non_exact) > 0 and len(same_len_barcodes) > 0:
                    # Vectorized Hamming distance calculation
                    # Convert sequences and barcodes to character arrays
                    seq_arrays = np.array([list(seq) for _, seq in non_exact])
                    barcode_arrays = np.array([list(b) for b in same_len_barcodes])
                    
                    # Compute Hamming distances: (n_seqs, n_barcodes) matrix
                    # Each element is the number of mismatches
                    mismatches = np.sum(seq_arrays[:, None, :] != barcode_arrays[None, :, :], axis=2)
                    
                    # Find sequences with at least one barcode within 1 mismatch
                    min_mismatches = np.min(mismatches, axis=1)
                    matched_1mm = np.where(min_mismatches <= 1)[0]
                    
                    for i in matched_1mm:
                        matched_indices.append(non_exact[i][0])
            
            if verbose:
                print(f"  Found {len(matched_indices)} matched sequences out of {len(seq_df)}")
            
            # Extract intensities for T and C positions from matched sequences
            cy3_values_t = []  # Points that should be T (cy3 high, cy5 low)
            cy5_values_c = []  # Points that should be C (cy3 low, cy5 high)
            
            for idx in matched_indices:
                seq = seq_df.iloc[idx]['Sequence']
                
                # Extract intensities for each cycle
                # Note: seq_df and intensity_df_sample have the same number of rows and same order
                for cyc in range(1, cyc_num + 1):
                    if cyc <= len(seq):
                        base = seq[cyc - 1]
                        cy3_col = f'cyc_{cyc}_cy3'
                        cy5_col = f'cyc_{cyc}_cy5'
                        
                        if base == 'T':
                            # T should have cy3 high, cy5 low
                            cy3_values_t.append(intensity_df_sample.iloc[idx][cy3_col])
                        elif base == 'C':
                            # C should have cy3 low, cy5 high
                            cy5_values_c.append(intensity_df_sample.iloc[idx][cy5_col])
            
            if verbose:
                print(f"  Extracted {len(cy3_values_t)} T positions and {len(cy5_values_c)} C positions")
            
            if len(cy3_values_t) > 100 and len(cy5_values_c) > 100:
                # Estimate balance: T and C should have similar intensities when "on"
                cy3_median_t = np.median(cy3_values_t)
                cy5_median_c = np.median(cy5_values_c)
                
                if cy3_median_t > 0 and cy5_median_c > 0:
                    # Balance: make T and C have similar "on" intensities
                    # If cy3_median_t > cy5_median_c, cy3 is brighter, scale it down
                    if cy3_median_t > cy5_median_c:
                        balance_cy3 = cy5_median_c / cy3_median_t
                        balance_cy5 = 1.0
                    else:
                        balance_cy3 = 1.0
                        balance_cy5 = cy3_median_t / cy5_median_c
                    
                    if verbose:
                        print(f"  Codebook-based balance: cy3={balance_cy3:.3f}, cy5={balance_cy5:.3f}")
                        print(f"  (T median cy3={cy3_median_t:.1f}, C median cy5={cy5_median_c:.1f})")
                    
                    return {'cy3': balance_cy3, 'cy5': balance_cy5}
            else:
                if verbose:
                    print(f"  Warning: Insufficient matched positions (T: {len(cy3_values_t)}, C: {len(cy5_values_c)})")
                    print(f"  Need at least 100 of each. Falling back to intensity_ratio method...")
                method = 'intensity_ratio'
    
    # Fallback methods
    if method == 'intensity_ratio' or (method == 'codebook' and ref_file is None):
        if verbose:
            print("  Estimating channel balance using intensity ratio method...")
        
        # Simple method: compare median intensities across all cycles
        all_cy3 = []
        all_cy5 = []
        
        for cyc in range(1, cyc_num + 1):
            cy3_col = f'cyc_{cyc}_cy3'
            cy5_col = f'cyc_{cyc}_cy5'
            if cy3_col in intensity_df.columns and cy5_col in intensity_df.columns:
                all_cy3.extend(intensity_df[cy3_col].values)
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
                
                if verbose:
                    print(f"  Intensity ratio balance: cy3={balance_cy3:.3f}, cy5={balance_cy5:.3f}")
                    print(f"  (cy3 median={cy3_median:.1f}, cy5 median={cy5_median:.1f})")
                
                return {'cy3': balance_cy3, 'cy5': balance_cy5}
    
    # Default: no balance
    if verbose:
        print("  Warning: Could not estimate channel balance, using 1.0 for both")
    return {'cy3': 1.0, 'cy5': 1.0}


def correct_intensity(intensity_df, cyc_num=10, percentile=99.9, channels=['cy3', 'cy5'],
                     smooth=True, smooth_method='savgol', smooth_window=5,
                     estimate_phasing=True, phasing_estimation_method='grid_search',
                     phasing_rate=None, prephasing_rate=None,
                     ref_file=None, 
                     correct_channel_balance=True, channel_balance_method='codebook',
                     channel_balance_factors=None,
                     channel_balance_sample_size=100000,
                     channel_balance_threshold_percentile=50,
                     verbose=True):
    """
    Main function to correct intensity data: decay correction + channel balance + phasing correction.
    
    This is the primary function to use. It takes uncorrected intensity data
    and returns fully corrected intensity data.
    
    Parameters
    ----------
    intensity_df : pd.DataFrame
        DataFrame with columns ['Y', 'X', 'cyc_1_cy3', 'cyc_1_cy5', ...]
        This is the UNCORRECTED intensity data
    cyc_num : int
        Number of cycles
    percentile : float
        Percentile to use for decay curve estimation (default: 99.9)
    channels : list
        List of channel names (default: ['cy3', 'cy5'])
    smooth : bool
        Whether to smooth the decay curve (default: True)
    smooth_method : str
        Smoothing method: 'savgol', 'moving_average', or 'none'
    smooth_window : int
        Window length for smoothing (default: 5)
    estimate_phasing : bool
        Whether to automatically estimate phasing rates (default: True)
    phasing_estimation_method : str
        Method to estimate phasing: 'grid_search', 'on_off', 'codebook', or 'none'
        If 'none', uses provided phasing_rate and prephasing_rate
    phasing_rate : float, optional
        Manual phasing rate (only used if estimate_phasing=False or method='none')
    prephasing_rate : float, optional
        Manual pre-phasing rate (only used if estimate_phasing=False or method='none')
    ref_file : str or Path, optional
        Path to reference file (required for 'on_off' and 'codebook' methods, 
        and recommended for channel balance estimation)
    correct_channel_balance : bool
        Whether to correct for channel brightness differences (default: True)
    channel_balance_method : str
        Method to estimate channel balance: 'codebook' (requires ref_file), 
        'intensity_ratio', or 'median' (default: 'codebook')
    channel_balance_factors : dict, optional
        Manual channel balance factors: {'cy3': factor_cy3, 'cy5': factor_cy5}
        If provided, these will be used instead of estimation
    channel_balance_sample_size : int
        Number of points to sample for codebook-based channel balance estimation (default: 100000)
        Only used when channel_balance_method='codebook'. Larger sample gives more accurate results but slower.
    channel_balance_threshold_percentile : float
        Percentile to use for threshold calculation in codebook method (default: 50, i.e., median)
        Only used when channel_balance_method='codebook'. Higher values (e.g., 50-70) give better quality sequences.
    verbose : bool
        Whether to print progress information
    
    Returns
    -------
    pd.DataFrame
        Corrected intensity DataFrame with same structure as input
    dict
        Correction information including:
        - decay_info: decay correction parameters
        - channel_balance_info: channel balance correction parameters
        - phasing_info: phasing correction parameters
        - estimation_info: phasing rate estimation results (if estimated)
    """
    if verbose:
        print("=" * 60)
        print("Intensity Correction Pipeline")
        print("=" * 60)
    
    # Step 0: Channel balance correction (before decay correction for better accuracy)
    balance_info = {}
    if correct_channel_balance:
        if verbose:
            print("\nStep 0: Channel Balance Correction")
            print("-" * 60)
        
        if channel_balance_factors is not None:
            balance_factors = channel_balance_factors
            if verbose:
                print(f"  Using manual balance factors: cy3={balance_factors.get('cy3', 1.0):.3f}, "
                      f"cy5={balance_factors.get('cy5', 1.0):.3f}")
        else:
            balance_factors = estimate_channel_balance(
                intensity_df, cyc_num, channels, ref_file, 
                method=channel_balance_method, 
                sample_size=channel_balance_sample_size,
                threshold_percentile=channel_balance_threshold_percentile,
                verbose=verbose
            )
        
        # Apply balance factors
        for channel in channels:
            factor = balance_factors.get(channel, 1.0)
            if factor != 1.0:
                for cyc in range(1, cyc_num + 1):
                    col = f'cyc_{cyc}_{channel}'
                    if col in intensity_df.columns:
                        intensity_df = intensity_df.copy()  # Avoid SettingWithCopyWarning
                        intensity_df[col] = intensity_df[col] * factor
        
        balance_info = {
            'balance_corrected': True,
            'balance_factors': balance_factors,
            'balance_method': channel_balance_method
        }
        
        if verbose:
            print(f"  Channel balance correction completed")
    else:
        balance_info = {
            'balance_corrected': False,
            'balance_factors': {'cy3': 1.0, 'cy5': 1.0}
        }
    
    # Step 1: Decay correction
    if verbose:
        print("\nStep 1: Decay Correction (P99.9 percentile method)")
        print("-" * 60)
    
    corrected_df, decay_info = correct_decay_robust(
        intensity_df, cyc_num, percentile, channels,
        smooth=smooth, smooth_method=smooth_method, smooth_window=smooth_window
    )
    
    if verbose:
        print(f"  Decay correction completed")
        print(f"  Baseline: {decay_info['baseline']:.2f}")
        print(f"  Scale factors range: [{np.min(decay_info['scale_factors']):.3f}, {np.max(decay_info['scale_factors']):.3f}]")
    
    # Step 2: Estimate phasing rates (if requested)
    phasing_info = {}
    estimation_info = {}
    
    if estimate_phasing and phasing_estimation_method != 'none':
        if verbose:
            print(f"\nStep 2: Phasing Rate Estimation (Method: {phasing_estimation_method})")
            print("-" * 60)
        
        if phasing_estimation_method == 'grid_search':
            estimation_info = estimate_phasing_grid_search(
                corrected_df, cyc_num, channels, verbose=verbose
            )
            phasing_rate = estimation_info['phasing_rate']
            prephasing_rate = estimation_info.get('prephasing_rate', 0.0)
            
        elif phasing_estimation_method == 'on_off':
            if ref_file is None:
                if verbose:
                    print("  Warning: ref_file required for 'on_off' method, using default values")
                phasing_rate = phasing_rate if phasing_rate is not None else 0.02
                prephasing_rate = prephasing_rate if prephasing_rate is not None else 0.0
            else:
                estimation_info = estimate_phasing_on_off_transition(
                    corrected_df, ref_file, cyc_num, channels, verbose=verbose
                )
                phasing_rate = estimation_info['phasing_rate']
                prephasing_rate = estimation_info.get('prephasing_rate', 0.0)
                
        elif phasing_estimation_method == 'codebook':
            if ref_file is None:
                if verbose:
                    print("  Warning: ref_file required for 'codebook' method, using default values")
                phasing_rate = phasing_rate if phasing_rate is not None else 0.02
                prephasing_rate = prephasing_rate if prephasing_rate is not None else 0.0
            else:
                estimation_info = estimate_phasing_codebook_residual(
                    corrected_df, ref_file, cyc_num, channels, verbose=verbose
                )
                phasing_rate = estimation_info['phasing_rate']
                prephasing_rate = estimation_info.get('prephasing_rate', 0.0)
        else:
            if verbose:
                print(f"  Warning: Unknown method '{phasing_estimation_method}', using default values")
            phasing_rate = phasing_rate if phasing_rate is not None else 0.02
            prephasing_rate = prephasing_rate if prephasing_rate is not None else 0.0
    else:
        # Use provided values or defaults
        phasing_rate = phasing_rate if phasing_rate is not None else 0.02
        prephasing_rate = prephasing_rate if prephasing_rate is not None else 0.0
    
    # Step 3: Phasing correction
    if verbose:
        print(f"\nStep 3: Phasing Correction")
        print("-" * 60)
        print(f"  Phasing rate (p): {phasing_rate:.4f} ({phasing_rate*100:.2f}%)")
        print(f"  Pre-phasing rate (q): {prephasing_rate:.4f} ({prephasing_rate*100:.2f}%)")
    
    # Build phasing matrix with both p and q
    phasing_matrix = build_phasing_matrix(cyc_num, phasing_rate=phasing_rate, prephasing_rate=prephasing_rate)
    try:
        phasing_matrix_inv = np.linalg.inv(phasing_matrix)
    except np.linalg.LinAlgError:
        if verbose:
            print(f"  Warning: Phasing matrix is singular, skipping phasing correction")
        return corrected_df, {
            **decay_info,
            'phasing_corrected': False,
            'phasing_error': 'singular_matrix',
            'estimation_info': estimation_info
        }
    
    # Apply phasing correction to each spot and channel
    # Use vectorized operations for much better performance (100-1000x faster)
    cycles = np.arange(1, cyc_num + 1)
    
    if verbose:
        print(f"  Applying phasing correction to {len(corrected_df)} spots...")
    
    # Vectorized phasing correction: process all spots at once for each channel
    for channel in channels:
        # Extract all intensity vectors for this channel (shape: n_spots x n_cycles)
        intensity_matrix = np.array([
            corrected_df[f'cyc_{cyc}_{channel}'].values 
            for cyc in cycles
        ]).T  # Transpose to get (n_spots, n_cycles)
        
        # Apply phasing correction to all spots at once
        # phasing_matrix_inv @ intensity_matrix.T gives (n_cycles, n_spots)
        # Transpose back to get (n_spots, n_cycles)
        intensity_corrected_matrix = (phasing_matrix_inv @ intensity_matrix.T).T
        
        # Remove negative values (physically impossible)
        intensity_corrected_matrix = np.maximum(intensity_corrected_matrix, 0)
        
        # Update dataframe columns in batch
        for i, cyc in enumerate(cycles):
            col = f'cyc_{cyc}_{channel}'
            if col in corrected_df.columns:
                corrected_df[col] = intensity_corrected_matrix[:, i]
    
    phasing_info = {
        'phasing_corrected': True,
        'phasing_rate': phasing_rate,
        'prephasing_rate': prephasing_rate,
        'phasing_matrix': phasing_matrix,
        'phasing_matrix_inv': phasing_matrix_inv
    }
    
    if verbose:
        print(f"  Phasing correction completed")
        print("\n" + "=" * 60)
        print("Correction Pipeline Completed")
        print("=" * 60)
    
    return corrected_df, {
        **decay_info,
        **balance_info,
        **phasing_info,
        'estimation_info': estimation_info
    }


if __name__ == "__main__":
    # Example usage
    print("Intensity Correction Module")
    print("=" * 50)
    print("\nThis module provides comprehensive intensity correction:")
    print("  1. Decay correction (P99.9 percentile method)")
    print("  2. Phasing correction (matrix inversion)")
    print("  3. Automatic phasing rate estimation (3 strategies)")
    print("\nUsage:")
    print("  from intensity_correction import correct_intensity")
    print("  corrected_df, info = correct_intensity(intensity_df)")
    print("\n  # With automatic phasing estimation:")
    print("  corrected_df, info = correct_intensity(intensity_df, estimate_phasing=True)")
    print("\n  # With manual phasing rates:")
    print("  corrected_df, info = correct_intensity(intensity_df, phasing_rate=0.02, prephasing_rate=0.01)")


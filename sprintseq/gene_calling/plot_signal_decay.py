"""
QC script to visualize signal decay across cycles.

This script follows the recommendations in info.md:
1. Plot total brightness from Cycle 1 to Cycle 10 for all spots
2. Check if the line looks like a "slide" (decreasing)
3. If Cycle 10 brightness is below 20% of Cycle 1, suggest truncating to first 8 cycles
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


BASE_DEST_DIRECTORY = Path(r'\\10.10.10.1\NAS Processed Images')
CHANNELS = ['cy3', 'cy5']
SEQ_CYCLE = 10


def calculate_cycle_brightness(intensity_df, cyc_num=SEQ_CYCLE, method='mean', percentile=None):
    """
    Calculate total brightness for each cycle across all spots.
    
    Parameters
    ----------
    intensity_df : pd.DataFrame
        DataFrame with columns ['cyc_1_cy3', 'cyc_1_cy5', ...]
    cyc_num : int
        Number of cycles (default: 10)
    method : str
        Method to aggregate across spots: 'mean', 'median', 'sum', 'total_mean', 'percentile'
        - 'mean': Mean of (cy3 + cy5) for each cycle
        - 'median': Median of (cy3 + cy5) for each cycle
        - 'sum': Sum of (cy3 + cy5) for each cycle
        - 'total_mean': Mean of all intensity values (cy3 and cy5 separately)
        - 'percentile': Percentile-based (requires percentile parameter, e.g., 99.9 for P99.9)
    percentile : float, optional
        Percentile value (0-100) when method='percentile'. Default: 99.9
    
    Returns
    -------
    dict
        Dictionary with keys:
        - 'cycles': array of cycle numbers [1, 2, ..., cyc_num]
        - 'total_brightness': array of total brightness per cycle
        - 'cy3_brightness': array of cy3 brightness per cycle
        - 'cy5_brightness': array of cy5 brightness per cycle
        - 'brightness_ratio': array of cycle N / cycle 1 ratio
        - 'percentile_used': percentile value if method='percentile'
    """
    cycles = np.arange(1, cyc_num + 1)
    total_brightness = []
    cy3_brightness = []
    cy5_brightness = []
    
    for cyc in cycles:
        cy3_col = f'cyc_{cyc}_cy3'
        cy5_col = f'cyc_{cyc}_cy5'
        
        if cy3_col not in intensity_df.columns or cy5_col not in intensity_df.columns:
            total_brightness.append(0)
            cy3_brightness.append(0)
            cy5_brightness.append(0)
            continue
        
        cy3_values = intensity_df[cy3_col].fillna(0).values
        cy5_values = intensity_df[cy5_col].fillna(0).values
        
        # Calculate per-spot total (cy3 + cy5)
        spot_totals = cy3_values + cy5_values
        
        if method == 'mean':
            total_brightness.append(np.mean(spot_totals))
            cy3_brightness.append(np.mean(cy3_values))
            cy5_brightness.append(np.mean(cy5_values))
        elif method == 'median':
            total_brightness.append(np.median(spot_totals))
            cy3_brightness.append(np.median(cy3_values))
            cy5_brightness.append(np.median(cy5_values))
        elif method == 'sum':
            total_brightness.append(np.sum(spot_totals))
            cy3_brightness.append(np.sum(cy3_values))
            cy5_brightness.append(np.sum(cy5_values))
        elif method == 'total_mean':
            # Mean of all intensity values (cy3 and cy5 separately)
            total_brightness.append((np.mean(cy3_values) + np.mean(cy5_values)))
            cy3_brightness.append(np.mean(cy3_values))
            cy5_brightness.append(np.mean(cy5_values))
        elif method == 'percentile':
            # Percentile-based method (robust to base composition bias)
            if percentile is None:
                percentile = 99.9
            # Use max channel intensity per spot (more robust)
            max_channel_per_spot = np.maximum(cy3_values, cy5_values)
            total_brightness.append(np.percentile(max_channel_per_spot, percentile))
            cy3_brightness.append(np.percentile(cy3_values, percentile))
            cy5_brightness.append(np.percentile(cy5_values, percentile))
        else:
            raise ValueError(f"Unknown method: {method}")
    
    total_brightness = np.array(total_brightness)
    cy3_brightness = np.array(cy3_brightness)
    cy5_brightness = np.array(cy5_brightness)
    
    # Calculate ratio relative to cycle 1
    # For percentile method, use max value as baseline (as recommended in info_percentile.md)
    if method == 'percentile':
        baseline = np.max(total_brightness)  # Use max as baseline
    else:
        baseline = total_brightness[0] if total_brightness[0] > 0 else 1.0
    
    if baseline > 0:
        brightness_ratio = total_brightness / baseline
    else:
        brightness_ratio = np.zeros_like(total_brightness)
    
    result = {
        'cycles': cycles,
        'total_brightness': total_brightness,
        'cy3_brightness': cy3_brightness,
        'cy5_brightness': cy5_brightness,
        'brightness_ratio': brightness_ratio
    }
    
    if method == 'percentile':
        result['percentile_used'] = percentile
        result['baseline'] = baseline
    
    return result


def plot_signal_decay(intensity_df, output_path=None, cyc_num=SEQ_CYCLE, 
                     method='mean', title_suffix=''):
    """
    Plot signal decay across cycles.
    
    Parameters
    ----------
    intensity_df : pd.DataFrame
        Intensity DataFrame
    output_path : str or Path, optional
        Path to save the plot. If None, display interactively.
    cyc_num : int
        Number of cycles
    method : str
        Aggregation method (see calculate_cycle_brightness)
    title_suffix : str
        Additional text to add to plot title
    """
    # Calculate brightness
    brightness_data = calculate_cycle_brightness(intensity_df, cyc_num, method)
    
    cycles = brightness_data['cycles']
    total_brightness = brightness_data['total_brightness']
    cy3_brightness = brightness_data['cy3_brightness']
    cy5_brightness = brightness_data['cy5_brightness']
    brightness_ratio = brightness_data['brightness_ratio']
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: Absolute brightness
    ax1.plot(cycles, total_brightness, 'o-', label='Total (cy3+cy5)', linewidth=2, markersize=8)
    ax1.plot(cycles, cy3_brightness, 's-', label='cy3', linewidth=1.5, markersize=6, alpha=0.7)
    ax1.plot(cycles, cy5_brightness, '^-', label='cy5', linewidth=1.5, markersize=6, alpha=0.7)
    ax1.set_xlabel('Cycle', fontsize=12)
    ax1.set_ylabel('Brightness', fontsize=12)
    ax1.set_title(f'Signal Decay Across Cycles{title_suffix}', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(cycles)
    
    # Plot 2: Relative brightness (normalized to cycle 1)
    ax2.plot(cycles, brightness_ratio, 'o-', label='Total (cy3+cy5)', linewidth=2, markersize=8, color='C0')
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Cycle 1 level')
    ax2.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='20% threshold')
    ax2.set_xlabel('Cycle', fontsize=12)
    ax2.set_ylabel('Brightness Ratio (Cycle N / Cycle 1)', fontsize=12)
    ax2.set_title('Relative Signal Decay (Normalized to Cycle 1)', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(cycles)
    ax2.set_ylim([0, max(1.1, np.max(brightness_ratio) * 1.1)])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    
    return brightness_data


def plot_spot_distributions_across_cycles(intensity_df, output_path, cyc_num=SEQ_CYCLE, title_suffix=''):
    """
    Plot per-spot (across all signal points) distributions across cycles,
    analogous to the aggregated tile plots but at spot level.
    
    - Boxplot of per-spot total intensity (cy3+cy5) per cycle
    - P99.9 curve across cycles
    - Boxplot of per-spot brightness ratio (cycle N / cycle 1) per cycle
    - Histogram of per-spot ratio at last cycle
    """
    cycles = np.arange(1, cyc_num + 1)
    all_brightness = []  # list of arrays, one per cycle (spots)
    
    for cyc in cycles:
        cy3_col = f'cyc_{cyc}_cy3'
        cy5_col = f'cyc_{cyc}_cy5'
        if cy3_col not in intensity_df.columns or cy5_col not in intensity_df.columns:
            all_brightness.append(np.array([]))
            continue
        vals = (
            intensity_df[cy3_col].fillna(0).values
            + intensity_df[cy5_col].fillna(0).values
        )
        all_brightness.append(vals)
    
    # Convert to 2D array with shape (n_spots, n_cycles) where possible
    # Use only cycles that have non-empty data and same length
    valid_lengths = [len(v) for v in all_brightness if len(v) > 0]
    if len(valid_lengths) == 0:
        return
    n_spots = max(valid_lengths)
    
    # Pad shorter cycles with NaN so we can compute ratios safely
    brightness_matrix = np.full((n_spots, len(cycles)), np.nan, dtype=float)
    for i, vals in enumerate(all_brightness):
        if len(vals) == 0:
            continue
        length = min(len(vals), n_spots)
        brightness_matrix[:length, i] = vals[:length]
    
    # Baseline = cycle 1 per-spot brightness
    baseline = brightness_matrix[:, 0]
    # Avoid division by zero: mask where baseline <= 0
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_matrix = np.where(
            baseline[:, None] > 0,
            brightness_matrix / baseline[:, None],
            np.nan,
        )
    
    # P99.9 per cycle
    p999_per_cycle = []
    for i in range(len(cycles)):
        col = brightness_matrix[:, i]
        col = col[~np.isnan(col)]
        if col.size == 0:
            p999_per_cycle.append(np.nan)
        else:
            p999_per_cycle.append(float(np.percentile(col, 99.9)))
    p999_per_cycle = np.array(p999_per_cycle)
    
    # Prepare figure
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    ax1, ax2, ax3 = axes
    
    # Plot 1: boxplot of brightness per cycle with P99.9 line
    box_data_brightness = []
    for i in range(len(cycles)):
        col = brightness_matrix[:, i]
        col = col[~np.isnan(col)]
        if col.size == 0:
            col = np.array([0.0])
        box_data_brightness.append(col)
    
    bp = ax1.boxplot(box_data_brightness, positions=cycles, widths=0.6, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    ax1.plot(cycles, p999_per_cycle, 's-', color='C2', label='P99.9 (per-spot)')
    ax1.set_xlabel('Cycle')
    ax1.set_ylabel('Per-spot total intensity (cy3+cy5)')
    ax1.set_title(f'Per-spot brightness distribution per cycle{title_suffix}')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xticks(cycles)
    ax1.legend(loc='best')
    
    # Plot 2: boxplot of per-spot brightness ratio per cycle
    box_data_ratio = []
    for i in range(len(cycles)):
        col = ratio_matrix[:, i]
        col = col[~np.isnan(col)]
        if col.size == 0:
            col = np.array([0.0])
        box_data_ratio.append(col)
    
    bp2 = ax2.boxplot(box_data_ratio, positions=cycles, widths=0.6, patch_artist=True)
    for patch in bp2['boxes']:
        patch.set_facecolor('lightcoral')
        patch.set_alpha(0.7)
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Cycle 1 level')
    ax2.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='20% threshold')
    ax2.set_xlabel('Cycle')
    ax2.set_ylabel('Per-spot brightness ratio (Cycle N / Cycle 1)')
    ax2.set_title('Per-spot brightness ratio distribution per cycle')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(cycles)
    ax2.set_ylim([0, max(1.1, np.nanmax(ratio_matrix) * 1.1)])
    ax2.legend(loc='best')
    
    # Plot 3: histogram of final-cycle per-spot ratio
    last_col = ratio_matrix[:, -1]
    last_col = last_col[~np.isnan(last_col)]
    if last_col.size == 0:
        last_col = np.array([0.0])
    ax3.hist(last_col, bins=40, edgecolor='black', alpha=0.7, color='C0')
    ax3.axvline(x=0.2, color='red', linestyle='--', linewidth=2, label='20% threshold')
    ax3.axvline(x=np.mean(last_col), color='blue', linestyle='-', linewidth=2, label=f'Mean: {np.mean(last_col):.1%}')
    ax3.axvline(x=np.median(last_col), color='green', linestyle='-', linewidth=2, label=f'Median: {np.median(last_col):.1%}')
    ax3.set_xlabel(f'Per-spot ratio at cycle {cyc_num} (Cycle {cyc_num} / Cycle 1)')
    ax3.set_ylabel('Number of spots')
    ax3.set_title(f'Per-spot final-cycle ratio distribution{title_suffix}')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend(loc='best')
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Per-spot distribution plot saved to: {output_path}")
    else:
        plt.show()


def analyze_signal_decay(intensity_df, cyc_num=SEQ_CYCLE, method='mean', 
                        warning_threshold=0.2):
    """
    Analyze signal decay and provide recommendations.
    
    Parameters
    ----------
    intensity_df : pd.DataFrame
        Intensity DataFrame
    cyc_num : int
        Number of cycles
    method : str
        Aggregation method
    warning_threshold : float
        Warning threshold (default: 0.2 = 20%)
    
    Returns
    -------
    dict
        Analysis results with recommendations
    """
    brightness_data = calculate_cycle_brightness(intensity_df, cyc_num, method)
    
    cycles = brightness_data['cycles']
    total_brightness = brightness_data['total_brightness']
    brightness_ratio = brightness_data['brightness_ratio']

    # ------------------------------------------------------------------
    # Per-spot distribution (across all signal points), not just mean.
    # For each cycle, we can look at the distribution of per-spot total
    # intensity (cy3 + cy5) and summarize it by mean / std / percentiles
    # including a high percentile such as P99.9.
    # ------------------------------------------------------------------
    def _cycle_spot_stats(cyc):
        cy3_col = f'cyc_{cyc}_cy3'
        cy5_col = f'cyc_{cyc}_cy5'
        if cy3_col not in intensity_df.columns or cy5_col not in intensity_df.columns:
            return None
        vals = (
            intensity_df[cy3_col].fillna(0).values
            + intensity_df[cy5_col].fillna(0).values
        )
        if vals.size == 0:
            return None
        mean_v = float(np.mean(vals))
        std_v = float(np.std(vals))
        p10 = float(np.percentile(vals, 10))
        p50 = float(np.percentile(vals, 50))
        p90 = float(np.percentile(vals, 90))
        p999 = float(np.percentile(vals, 99.9))
        return {
            "mean": mean_v,
            "std": std_v,
            "p10": p10,
            "p50": p50,
            "p90": p90,
            "p999": p999,
        }

    spot_dist_cycle1 = _cycle_spot_stats(1)
    spot_dist_last = _cycle_spot_stats(cyc_num)
    
    # Calculate decay rate (exponential fit)
    # I(n) = I(1) * E^(n-1)
    # log(I(n)/I(1)) = (n-1) * log(E)
    if total_brightness[0] > 0 and len(cycles) > 1:
        # Use cycles 1-5 for fitting (more stable)
        fit_cycles = cycles[:min(5, len(cycles))]
        fit_ratios = brightness_ratio[:len(fit_cycles)]
        fit_ratios = fit_ratios[fit_ratios > 0]
        fit_cycles = fit_cycles[:len(fit_ratios)]
        
        if len(fit_ratios) > 1:
            # Linear fit in log space
            log_ratios = np.log(fit_ratios)
            n_minus_1 = fit_cycles - 1
            if np.sum(n_minus_1**2) > 0:
                slope = np.sum(n_minus_1 * log_ratios) / np.sum(n_minus_1**2)
                efficiency = np.exp(slope)  # E
                decay_per_cycle = (1 - efficiency) * 100  # percentage
            else:
                efficiency = 1.0
                decay_per_cycle = 0.0
        else:
            efficiency = 1.0
            decay_per_cycle = 0.0
    else:
        efficiency = 1.0
        decay_per_cycle = 0.0
    
    # Check if cycle 10 is below threshold
    cycle_10_ratio = brightness_ratio[-1] if len(brightness_ratio) >= 10 else 0
    severe_decay = cycle_10_ratio < warning_threshold
    
    # Check if it looks like a "slide" (monotonically decreasing)
    is_decreasing = np.all(np.diff(total_brightness) <= 0)
    
    # Find where signal drops below threshold
    below_threshold_cycles = np.where(brightness_ratio < warning_threshold)[0]
    if len(below_threshold_cycles) > 0:
        first_below_threshold = cycles[below_threshold_cycles[0]]
    else:
        first_below_threshold = None
    
    # Recommendations
    recommendations = []
    if severe_decay:
        recommendations.append(
            f"⚠️  WARNING: Cycle {cyc_num} brightness ({cycle_10_ratio:.1%}) is below {warning_threshold:.0%} of Cycle 1."
        )
        recommendations.append(
            f"   → Consider truncating to first 8 cycles to avoid noise amplification."
        )
    elif cycle_10_ratio < 0.3:
        recommendations.append(
            f"⚠️  Cycle {cyc_num} brightness ({cycle_10_ratio:.1%}) is below 30% of Cycle 1."
        )
        recommendations.append(
            f"   → Signal correction may amplify noise in later cycles."
        )
    
    if is_decreasing:
        recommendations.append(
            "✓ Signal shows clear decay pattern (monotonically decreasing)."
        )
        recommendations.append(
            f"   → Estimated decay rate: {decay_per_cycle:.2f}% per cycle (Efficiency: {efficiency:.3f})"
        )
    else:
        recommendations.append(
            "⚠️  Signal does not show clear monotonic decay pattern."
        )
        recommendations.append(
            "   → May indicate other issues (phasing, background noise, etc.)"
        )
    
    if first_below_threshold:
        recommendations.append(
            f"   → Signal drops below {warning_threshold:.0%} threshold at Cycle {first_below_threshold}"
        )
    
    return {
        'brightness_data': brightness_data,
        'efficiency': efficiency,
        'decay_per_cycle': decay_per_cycle,
        'cycle_10_ratio': cycle_10_ratio,
        'severe_decay': severe_decay,
        'is_decreasing': is_decreasing,
        'first_below_threshold': first_below_threshold,
        'recommendations': recommendations,
        # Per-spot distribution summaries (across all signal points)
        'spot_dist_cycle1': spot_dist_cycle1,
        'spot_dist_last': spot_dist_last,
    }


def plot_signal_decay_from_file(intensity_file, output_path=None, 
                                cyc_num=SEQ_CYCLE, method='mean'):
    """
    Load intensity data from CSV file and plot signal decay.
    
    Parameters
    ----------
    intensity_file : str or Path
        Path to intensity CSV file (with columns like 'cyc_1_cy3', 'cyc_1_cy5', ...)
    output_path : str or Path, optional
        Path to save the plot
    cyc_num : int
        Number of cycles
    method : str
        Aggregation method
    """
    intensity_df = pd.read_csv(intensity_file)
    
    # Extract name from file path for title
    file_stem = Path(intensity_file).stem
    title_suffix = f' - {file_stem}'
    
    # 1) 数值分析：直接用 calculate_cycle_brightness（不画简单 2 子图）
    brightness_data = calculate_cycle_brightness(
        intensity_df,
        cyc_num=cyc_num,
        method=method,
    )

    # 2) 绘图：按“每个 spot 当成一个 tile”复刻 10 宫格大图
    if output_path is not None:
        output_path = Path(output_path)
        if output_path.is_dir():
            fig_path = output_path / f"{file_stem}_signal_decay_spots_aggregated.png"
        else:
            fig_path = output_path

        # 这里直接把所有 spot 当成 tile 传给 tile 聚合绘图函数
        # 先构造 (n_spots, n_cycles) 的 cy3 / cy5 / brightness / ratio 矩阵
        cycles = np.arange(1, cyc_num + 1)
        n_spots = len(intensity_df)
        n_cycles = len(cycles)

        cy3_matrix = np.zeros((n_spots, n_cycles), dtype=float)
        cy5_matrix = np.zeros((n_spots, n_cycles), dtype=float)
        for i, cyc in enumerate(cycles):
            cy3_col = f'cyc_{cyc}_cy3'
            cy5_col = f'cyc_{cyc}_cy5'
            if cy3_col not in intensity_df.columns or cy5_col not in intensity_df.columns:
                continue
            cy3_vals = intensity_df[cy3_col].fillna(0).values
            cy5_vals = intensity_df[cy5_col].fillna(0).values
            cy3_matrix[:, i] = cy3_vals[:n_spots]
            cy5_matrix[:, i] = cy5_vals[:n_spots]

        all_cy3_brightness = cy3_matrix
        all_cy5_brightness = cy5_matrix
        all_brightness = cy3_matrix + cy5_matrix  # (n_spots, n_cycles)

        baseline = all_brightness[:, 0]
        with np.errstate(divide='ignore', invalid='ignore'):
            all_ratios = np.where(
                baseline[:, None] > 0,
                all_brightness / baseline[:, None],
                0.0,
            )

        # 为了接口兼容，P99.9 版本这里复用同一个矩阵
        all_brightness_p999 = all_brightness.copy()
        all_ratios_p999 = all_ratios.copy()

        mean_brightness = np.mean(all_brightness, axis=0)
        median_brightness = np.median(all_brightness, axis=0)
        std_brightness = np.std(all_brightness, axis=0)

        mean_cy3 = np.mean(all_cy3_brightness, axis=0)
        median_cy3 = np.median(all_cy3_brightness, axis=0)
        std_cy3 = np.std(all_cy3_brightness, axis=0)

        mean_cy5 = np.mean(all_cy5_brightness, axis=0)
        median_cy5 = np.median(all_cy5_brightness, axis=0)
        std_cy5 = np.std(all_cy5_brightness, axis=0)

        mean_ratio = np.mean(all_ratios, axis=0)
        median_ratio = np.median(all_ratios, axis=0)
        std_ratio = np.std(all_ratios, axis=0)

        # “P99.9 版本”的统计，结构上保持接口一致
        mean_brightness_p999 = mean_brightness.copy()
        median_brightness_p999 = median_brightness.copy()
        mean_ratio_p999 = mean_ratio.copy()
        median_ratio_p999 = median_ratio.copy()

        plot_aggregated_signal_decay(
            cycles=cycles,
            all_brightness=all_brightness,
            all_ratios=all_ratios,
            all_cy3_brightness=all_cy3_brightness,
            all_cy5_brightness=all_cy5_brightness,
            all_brightness_p999=all_brightness_p999,
            all_ratios_p999=all_ratios_p999,
            mean_brightness=mean_brightness,
            median_brightness=median_brightness,
            std_brightness=std_brightness,
            mean_brightness_p999=mean_brightness_p999,
            median_brightness_p999=median_brightness_p999,
            mean_cy3=mean_cy3,
            median_cy3=median_cy3,
            std_cy3=std_cy3,
            mean_cy5=mean_cy5,
            median_cy5=median_cy5,
            std_cy5=std_cy5,
            mean_ratio=mean_ratio,
            median_ratio=median_ratio,
            std_ratio=std_ratio,
            mean_ratio_p999=mean_ratio_p999,
            median_ratio_p999=median_ratio_p999,
            output_path=fig_path,
            run_id=f"Per-spot{title_suffix}",
            n_tiles=n_spots,
        )
    
    # Analyze
    analysis = analyze_signal_decay(intensity_df, cyc_num=cyc_num, method=method)
    
    # Print recommendations
    print("\n" + "=" * 80)
    print("Signal Decay Analysis")
    print("=" * 80)
    print(f"Cycle 1 brightness: {brightness_data['total_brightness'][0]:.2f}")
    print(f"Cycle {cyc_num} brightness: {brightness_data['total_brightness'][-1]:.2f}")
    # Per-spot distribution across all signal points (cy3+cy5)
    if analysis.get('spot_dist_cycle1') is not None:
        d1 = analysis['spot_dist_cycle1']
        print(f"  Cycle 1 per-spot total intensity (cy3+cy5):")
        print(f"    mean ± std: {d1['mean']:.2f} ± {d1['std']:.2f}")
        print(f"    P10 / P50 / P90 / P99.9: {d1['p10']:.2f} / {d1['p50']:.2f} / {d1['p90']:.2f} / {d1['p999']:.2f}")
    if analysis.get('spot_dist_last') is not None:
        dl = analysis['spot_dist_last']
        print(f"  Cycle {cyc_num} per-spot total intensity (cy3+cy5):")
        print(f"    mean ± std: {dl['mean']:.2f} ± {dl['std']:.2f}")
        print(f"    P10 / P50 / P90 / P99.9: {dl['p10']:.2f} / {dl['p50']:.2f} / {dl['p90']:.2f} / {dl['p999']:.2f}")
    print(f"Cycle {cyc_num} / Cycle 1 ratio: {analysis['cycle_10_ratio']:.1%}")
    print(f"Estimated efficiency (E): {analysis['efficiency']:.3f}")
    print(f"Estimated decay per cycle: {analysis['decay_per_cycle']:.2f}%")
    print("\nRecommendations:")
    for rec in analysis['recommendations']:
        print(f"  {rec}")
    print("=" * 80 + "\n")
    
    return brightness_data, analysis


def aggregate_all_tiles(run_id, output_dir=None, cyc_num=SEQ_CYCLE, method='mean'):
    """
    Aggregate signal decay data from all tiles and plot distribution.
    
    Parameters
    ----------
    run_id : str
        Run ID (e.g., '20251128_ZCH_BZ09_Re2_mut_new')
    output_dir : str or Path, optional
        Directory to save plots. If None, use readout/tiles/ directory.
    cyc_num : int
        Number of cycles
    method : str
        Aggregation method
    
    Returns
    -------
    dict
        Aggregated data and analysis results
    """
    dest_directory = os.path.join(BASE_DEST_DIRECTORY, f'{run_id}_processed')
    read_directory = os.path.join(dest_directory, 'readout')
    tiles_output_directory = os.path.join(read_directory, 'tiles')
    
    # Try multiple file location patterns
    intensity_files = []
    
    # Pattern 1: Check for tile-specific files in readout/tiles/
    if os.path.exists(tiles_output_directory):
        intensity_files = sorted(Path(tiles_output_directory).glob('*_intensity_filtered.csv'))
        # Also try alternative naming patterns
        if len(intensity_files) == 0:
            intensity_files = sorted(Path(tiles_output_directory).glob('*_intensity_raw.csv'))
        if len(intensity_files) == 0:
            intensity_files = sorted(Path(tiles_output_directory).glob('*intensity*.csv'))
    
    # Pattern 2: Check for single intensity.csv in readout/ directory (new format)
    if len(intensity_files) == 0 and os.path.exists(read_directory):
        single_intensity_file = Path(read_directory) / 'intensity.csv'
        if single_intensity_file.exists():
            # If single file exists, treat it as one "tile" for aggregation
            intensity_files = [single_intensity_file]
            print(f"Found single intensity file: {single_intensity_file}")
            print("Note: This appears to be a new format with a single intensity.csv file.")
            print("      Consider using --intensity_file mode instead for per-spot analysis.")
    
    if len(intensity_files) == 0:
        # Provide detailed error message
        error_msg = f"No intensity files found.\n"
        error_msg += f"  Searched in: {tiles_output_directory}\n"
        if os.path.exists(tiles_output_directory):
            all_files = list(Path(tiles_output_directory).glob('*.csv'))
            if len(all_files) > 0:
                error_msg += f"  Found {len(all_files)} CSV files (but not matching intensity pattern):\n"
                for f in all_files[:10]:  # Show first 10
                    error_msg += f"    - {f.name}\n"
                if len(all_files) > 10:
                    error_msg += f"    ... and {len(all_files) - 10} more files\n"
            else:
                error_msg += f"  Directory exists but contains no CSV files.\n"
        else:
            error_msg += f"  Directory does not exist.\n"
        
        # Also check readout/ directory
        if os.path.exists(read_directory):
            readout_files = list(Path(read_directory).glob('*.csv'))
            if len(readout_files) > 0:
                error_msg += f"\n  Found {len(readout_files)} CSV files in readout/ directory:\n"
                for f in readout_files[:10]:
                    error_msg += f"    - {f.name}\n"
                if len(readout_files) > 10:
                    error_msg += f"    ... and {len(readout_files) - 10} more files\n"
        
        error_msg += f"\n  Tip: If you have a single intensity.csv file, use --intensity_file mode instead."
        raise ValueError(error_msg)
    
    if output_dir is None:
        output_dir = tiles_output_directory if os.path.exists(tiles_output_directory) else read_directory
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Found {len(intensity_files)} intensity files. Aggregating data...")
    
    # Collect data from all tiles
    all_brightness_data = []
    all_ratios = []
    all_efficiencies = []
    all_cycle10_ratios = []
    all_brightness_data_percentile = []  # Percentile-based data
    all_ratios_percentile = []
    tile_names = []
    
    for intensity_file in intensity_files:
        try:
            intensity_df = pd.read_csv(intensity_file)
            if len(intensity_df) == 0:
                continue
            
            # Extract tile name - handle different naming patterns
            tile_stem = intensity_file.stem
            if '_intensity_filtered' in tile_stem:
                tile_stem = tile_stem.replace('_intensity_filtered', '')
            elif '_intensity_raw' in tile_stem:
                tile_stem = tile_stem.replace('_intensity_raw', '')
            elif tile_stem == 'intensity':
                tile_stem = 'all_spots'  # Single file format
            tile_names.append(tile_stem)
            
            # Calculate brightness for this tile (using specified method)
            brightness_data = calculate_cycle_brightness(intensity_df, cyc_num, method)
            all_brightness_data.append(brightness_data)
            all_ratios.append(brightness_data['brightness_ratio'])
            
            # Also calculate percentile-based (for comparison)
            brightness_data_sample = calculate_cycle_brightness(intensity_df, cyc_num, 'percentile', percentile=99.9)
            all_brightness_data_percentile.append(brightness_data_sample)
            all_ratios_percentile.append(brightness_data_sample['brightness_ratio'])
            
            # Analyze this tile
            analysis = analyze_signal_decay(intensity_df, cyc_num, method)
            all_efficiencies.append(analysis['efficiency'])
            all_cycle10_ratios.append(analysis['cycle_10_ratio'])
            
        except Exception as e:
            print(f"Warning: Error processing {intensity_file}: {e}")
            continue
    
    if len(all_brightness_data) == 0:
        raise ValueError("No valid intensity data found in any tile")
    
    print(f"Successfully processed {len(all_brightness_data)} tiles.")
    
    # Aggregate data
    cycles = all_brightness_data[0]['cycles']
    all_total_brightness = np.array([bd['total_brightness'] for bd in all_brightness_data])
    all_cy3_brightness = np.array([bd['cy3_brightness'] for bd in all_brightness_data])
    all_cy5_brightness = np.array([bd['cy5_brightness'] for bd in all_brightness_data])
    all_ratios_array = np.array(all_ratios)
    
    # Percentile-based data
    all_total_brightness_p999 = np.array([bd['total_brightness'] for bd in all_brightness_data_percentile])
    all_ratios_array_p999 = np.array(all_ratios_percentile)
    
    # Calculate statistics across tiles
    mean_brightness = np.mean(all_total_brightness, axis=0)
    median_brightness = np.median(all_total_brightness, axis=0)
    std_brightness = np.std(all_total_brightness, axis=0)
    
    mean_brightness_p999 = np.mean(all_total_brightness_p999, axis=0)
    median_brightness_p999 = np.median(all_total_brightness_p999, axis=0)
    
    mean_cy3 = np.mean(all_cy3_brightness, axis=0)
    median_cy3 = np.median(all_cy3_brightness, axis=0)
    std_cy3 = np.std(all_cy3_brightness, axis=0)
    
    mean_cy5 = np.mean(all_cy5_brightness, axis=0)
    median_cy5 = np.median(all_cy5_brightness, axis=0)
    std_cy5 = np.std(all_cy5_brightness, axis=0)
    
    mean_ratio = np.mean(all_ratios_array, axis=0)
    median_ratio = np.median(all_ratios_array, axis=0)
    std_ratio = np.std(all_ratios_array, axis=0)
    
    mean_ratio_p999 = np.mean(all_ratios_array_p999, axis=0)
    median_ratio_p999 = np.median(all_ratios_array_p999, axis=0)
    
    # Plot aggregated distribution
    output_path = os.path.join(output_dir, f'{run_id}_signal_decay_aggregated.png')
    plot_aggregated_signal_decay(
        cycles=cycles,
        all_brightness=all_total_brightness,
        all_ratios=all_ratios_array,
        all_cy3_brightness=all_cy3_brightness,
        all_cy5_brightness=all_cy5_brightness,
        all_brightness_p999=all_total_brightness_p999,
        all_ratios_p999=all_ratios_array_p999,
        mean_brightness=mean_brightness,
        median_brightness=median_brightness,
        std_brightness=std_brightness,
        mean_brightness_p999=mean_brightness_p999,
        median_brightness_p999=median_brightness_p999,
        mean_cy3=mean_cy3,
        median_cy3=median_cy3,
        std_cy3=std_cy3,
        mean_cy5=mean_cy5,
        median_cy5=median_cy5,
        std_cy5=std_cy5,
        mean_ratio=mean_ratio,
        median_ratio=median_ratio,
        std_ratio=std_ratio,
        mean_ratio_p999=mean_ratio_p999,
        median_ratio_p999=median_ratio_p999,
        output_path=output_path,
        run_id=run_id,
        n_tiles=len(all_brightness_data)
    )
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("Aggregated Signal Decay Analysis (All Tiles)")
    print("=" * 80)
    print(f"Number of tiles analyzed: {len(all_brightness_data)}")
    print(f"\nCycle 1 brightness:")
    print(f"  Mean: {mean_brightness[0]:.2f} ± {std_brightness[0]:.2f}")
    print(f"  Median: {median_brightness[0]:.2f}")
    print(f"  Range: [{np.min(all_total_brightness[:, 0]):.2f}, {np.max(all_total_brightness[:, 0]):.2f}]")
    print(f"\nCycle {cyc_num} brightness:")
    print(f"  Mean: {mean_brightness[-1]:.2f} ± {std_brightness[-1]:.2f}")
    print(f"  Median: {median_brightness[-1]:.2f}")
    print(f"  Range: [{np.min(all_total_brightness[:, -1]):.2f}, {np.max(all_total_brightness[:, -1]):.2f}]")
    print(f"\nCycle {cyc_num} / Cycle 1 ratio:")
    print(f"  Mean: {np.mean(all_cycle10_ratios):.1%} ± {np.std(all_cycle10_ratios):.1%}")
    print(f"  Median: {np.median(all_cycle10_ratios):.1%}")
    print(f"  Range: [{np.min(all_cycle10_ratios):.1%}, {np.max(all_cycle10_ratios):.1%}]")
    print(f"\nEstimated efficiency (E):")
    print(f"  Mean: {np.mean(all_efficiencies):.3f} ± {np.std(all_efficiencies):.3f}")
    print(f"  Median: {np.median(all_efficiencies):.3f}")
    
    # Count tiles with severe decay
    severe_decay_count = np.sum(np.array(all_cycle10_ratios) < 0.2)
    moderate_decay_count = np.sum((np.array(all_cycle10_ratios) >= 0.2) & 
                                   (np.array(all_cycle10_ratios) < 0.3))
    good_signal_count = len(all_cycle10_ratios) - severe_decay_count - moderate_decay_count
    
    print(f"\nSignal quality distribution:")
    print(f"  Good signal (≥30%): {good_signal_count} tiles ({good_signal_count/len(all_cycle10_ratios):.1%})")
    print(f"  Moderate decay (20-30%): {moderate_decay_count} tiles ({moderate_decay_count/len(all_cycle10_ratios):.1%})")
    print(f"  Severe decay (<20%): {severe_decay_count} tiles ({severe_decay_count/len(all_cycle10_ratios):.1%})")
    
    if severe_decay_count > 0:
        print(f"\n⚠️  WARNING: {severe_decay_count} tiles have severe signal decay (<20%).")
        print(f"   → Consider truncating to first 8 cycles for these tiles.")
    
    print("=" * 80 + "\n")
    
    return {
        'n_tiles': len(all_brightness_data),
        'tile_names': tile_names,
        'mean_brightness': mean_brightness,
        'median_brightness': median_brightness,
        'std_brightness': std_brightness,
        'mean_ratio': mean_ratio,
        'median_ratio': median_ratio,
        'std_ratio': std_ratio,
        'all_efficiencies': all_efficiencies,
        'all_cycle10_ratios': all_cycle10_ratios,
        'severe_decay_count': severe_decay_count
    }


def plot_aggregated_signal_decay(cycles, all_brightness, all_ratios,
                                 all_cy3_brightness, all_cy5_brightness,
                                 all_brightness_p999, all_ratios_p999,
                                 mean_brightness, median_brightness, std_brightness,
                                 mean_brightness_p999, median_brightness_p999,
                                 mean_cy3, median_cy3, std_cy3,
                                 mean_cy5, median_cy5, std_cy5,
                                 mean_ratio, median_ratio, std_ratio,
                                 mean_ratio_p999, median_ratio_p999,
                                 output_path, run_id, n_tiles):
    """
    Plot aggregated signal decay with distribution across all tiles.
    
    Parameters
    ----------
    cycles : np.ndarray
        Cycle numbers
    all_brightness : np.ndarray
        Shape (n_tiles, n_cycles) - brightness for each tile
    all_ratios : np.ndarray
        Shape (n_tiles, n_cycles) - brightness ratio for each tile
    all_cy3_brightness : np.ndarray
        Shape (n_tiles, n_cycles) - cy3 brightness for each tile
    all_cy5_brightness : np.ndarray
        Shape (n_tiles, n_cycles) - cy5 brightness for each tile
    mean_brightness, median_brightness, std_brightness : np.ndarray
        Statistics across tiles
    mean_cy3, median_cy3, std_cy3 : np.ndarray
        Statistics for cy3 across tiles
    mean_cy5, median_cy5, std_cy5 : np.ndarray
        Statistics for cy5 across tiles
    mean_ratio, median_ratio, std_ratio : np.ndarray
        Statistics across tiles
    output_path : str
        Path to save plot
    run_id : str
        Run ID for title
    n_tiles : int
        Number of tiles
    """
    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(5, 2, hspace=0.35, wspace=0.3)
    
    # Plot 1: Mean vs P99.9 brightness comparison (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(cycles, mean_brightness, 'o-', label='Mean', linewidth=2, markersize=8, color='C0', alpha=0.7)
    ax1.fill_between(cycles, 
                     mean_brightness - std_brightness,
                     mean_brightness + std_brightness,
                     alpha=0.2, color='C0', label='Mean ±1 SD')
    ax1.plot(cycles, mean_brightness_p999, 's-', label='P99.9 (Robust)', linewidth=2, markersize=8, color='C2')
    ax1.plot(cycles, median_brightness_p999, '^--', label='P99.9 Median', linewidth=1.5, markersize=6, color='C2', alpha=0.7)
    ax1.set_xlabel('Cycle', fontsize=11)
    ax1.set_ylabel('Brightness', fontsize=11)
    ax1.set_title(f'Mean vs P99.9 Signal Decay (n={n_tiles} tiles)', fontsize=12, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(cycles)
    
    # Plot 2: Boxplot of brightness per cycle (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    box_data = [all_brightness[:, i] for i in range(len(cycles))]
    bp = ax2.boxplot(box_data, positions=cycles, widths=0.6, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    ax2.set_xlabel('Cycle', fontsize=11)
    ax2.set_ylabel('Brightness', fontsize=11)
    ax2.set_title(f'Brightness Distribution per Cycle (n={n_tiles} tiles)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(cycles)
    
    # Plot 3: Mean vs P99.9 ratio comparison (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(cycles, mean_ratio, 'o-', label='Mean Ratio', linewidth=2, markersize=8, color='C0', alpha=0.7)
    ax3.fill_between(cycles,
                     mean_ratio - std_ratio,
                     mean_ratio + std_ratio,
                     alpha=0.2, color='C0', label='Mean ±1 SD')
    ax3.plot(cycles, mean_ratio_p999, 's-', label='P99.9 Ratio (Robust)', linewidth=2, markersize=8, color='C2')
    ax3.plot(cycles, median_ratio_p999, '^--', label='P99.9 Median', linewidth=1.5, markersize=6, color='C2', alpha=0.7)
    ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    ax3.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='20% threshold')
    ax3.set_xlabel('Cycle', fontsize=11)
    ax3.set_ylabel('Brightness Ratio', fontsize=11)
    ax3.set_title('Mean vs P99.9 Relative Decay (P99.9 should be smoother)', fontsize=12, fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(cycles)
    ax3.set_ylim([0, max(1.1, max(np.max(mean_ratio + std_ratio), np.max(mean_ratio_p999)) * 1.1)])
    
    # Plot 4: Boxplot of ratios per cycle (middle right)
    ax4 = fig.add_subplot(gs[1, 1])
    box_data_ratio = [all_ratios[:, i] for i in range(len(cycles))]
    bp2 = ax4.boxplot(box_data_ratio, positions=cycles, widths=0.6, patch_artist=True)
    for patch in bp2['boxes']:
        patch.set_facecolor('lightcoral')
        patch.set_alpha(0.7)
    ax4.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax4.axhline(y=0.2, color='red', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Cycle', fontsize=11)
    ax4.set_ylabel('Brightness Ratio', fontsize=11)
    ax4.set_title('Ratio Distribution per Cycle', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_xticks(cycles)
    ax4.set_ylim([0, max(1.1, np.max(all_ratios) * 1.1)])
    
    # Plot 5: All tile curves (overlay, bottom left)
    ax5 = fig.add_subplot(gs[2, 0])
    for i, ratio in enumerate(all_ratios):
        ax5.plot(cycles, ratio, alpha=0.2, linewidth=0.5, color='gray')
    ax5.plot(cycles, mean_ratio, 'o-', label='Mean', linewidth=2, markersize=8, color='C0')
    ax5.plot(cycles, median_ratio, 's--', label='Median', linewidth=1.5, markersize=6, color='C1')
    ax5.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='20% threshold')
    ax5.set_xlabel('Cycle', fontsize=11)
    ax5.set_ylabel('Brightness Ratio', fontsize=11)
    ax5.set_title(f'All Tile Curves (n={n_tiles}, gray) + Mean/Median', fontsize=12, fontweight='bold')
    ax5.legend(loc='best')
    ax5.grid(True, alpha=0.3)
    ax5.set_xticks(cycles)
    ax5.set_ylim([0, max(1.1, np.max(all_ratios) * 1.1)])
    
    # Plot 6: Cycle 10 ratio distribution (bottom right)
    ax6 = fig.add_subplot(gs[2, 1])
    cycle10_ratios = all_ratios[:, -1]
    ax6.hist(cycle10_ratios, bins=20, edgecolor='black', alpha=0.7, color='C2')
    ax6.axvline(x=0.2, color='red', linestyle='--', linewidth=2, label='20% threshold')
    ax6.axvline(x=np.mean(cycle10_ratios), color='blue', linestyle='-', linewidth=2, label=f'Mean: {np.mean(cycle10_ratios):.1%}')
    ax6.axvline(x=np.median(cycle10_ratios), color='green', linestyle='-', linewidth=2, label=f'Median: {np.median(cycle10_ratios):.1%}')
    ax6.set_xlabel('Cycle 10 / Cycle 1 Ratio', fontsize=11)
    ax6.set_ylabel('Number of Tiles', fontsize=11)
    ax6.set_title('Cycle 10 Ratio Distribution', fontsize=12, fontweight='bold')
    ax6.legend(loc='best')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Plot 7: cy3 and cy5 separate intensity (new, bottom left)
    ax7 = fig.add_subplot(gs[3, 0])
    ax7.plot(cycles, mean_cy3, 'o-', label='cy3 Mean', linewidth=2, markersize=8, color='C3')
    ax7.fill_between(cycles,
                     mean_cy3 - std_cy3,
                     mean_cy3 + std_cy3,
                     alpha=0.3, color='C3', label='cy3 ±1 SD')
    ax7.plot(cycles, mean_cy5, 's-', label='cy5 Mean', linewidth=2, markersize=8, color='C4')
    ax7.fill_between(cycles,
                     mean_cy5 - std_cy5,
                     mean_cy5 + std_cy5,
                     alpha=0.3, color='C4', label='cy5 ±1 SD')
    ax7.plot(cycles, median_cy3, '--', label='cy3 Median', linewidth=1.5, markersize=6, color='C3', alpha=0.7)
    ax7.plot(cycles, median_cy5, '--', label='cy5 Median', linewidth=1.5, markersize=6, color='C4', alpha=0.7)
    ax7.set_xlabel('Cycle', fontsize=11)
    ax7.set_ylabel('Brightness', fontsize=11)
    ax7.set_title(f'cy3 and cy5 Intensity Decay (n={n_tiles} tiles)', fontsize=12, fontweight='bold')
    ax7.legend(loc='best', ncol=2)
    ax7.grid(True, alpha=0.3)
    ax7.set_xticks(cycles)
    
    # Plot 8: cy3 and cy5 boxplot comparison (new, bottom right)
    ax8 = fig.add_subplot(gs[3, 1])
    # Prepare data for boxplot: [cy3_cycle1, cy5_cycle1, cy3_cycle2, cy5_cycle2, ...]
    box_data_combined = []
    box_positions = []
    box_labels = []
    for i, cyc in enumerate(cycles):
        box_data_combined.append(all_cy3_brightness[:, i])
        box_positions.append(cyc - 0.2)
        box_labels.append(f'cy3\nC{cyc}')
        box_data_combined.append(all_cy5_brightness[:, i])
        box_positions.append(cyc + 0.2)
        box_labels.append(f'cy5\nC{cyc}')
    
    bp3 = ax8.boxplot(box_data_combined, positions=box_positions, widths=0.3, patch_artist=True)
    # Color boxes: cy3 in one color, cy5 in another
    for i, patch in enumerate(bp3['boxes']):
        if i % 2 == 0:  # cy3
            patch.set_facecolor('C3')
        else:  # cy5
            patch.set_facecolor('C4')
        patch.set_alpha(0.7)
    
    # Add cycle labels on x-axis
    ax8.set_xticks(cycles)
    ax8.set_xticklabels([f'C{cyc}' for cyc in cycles])
    ax8.set_xlabel('Cycle', fontsize=11)
    ax8.set_ylabel('Brightness', fontsize=11)
    ax8.set_title('cy3 vs cy5 Distribution per Cycle', fontsize=12, fontweight='bold')
    ax8.grid(True, alpha=0.3, axis='y')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='C3', alpha=0.7, label='cy3'),
        Patch(facecolor='C4', alpha=0.7, label='cy5')
    ]
    ax8.legend(handles=legend_elements, loc='best')
    
    # Plot 9: P99.9 decay curve (new, bottom row)
    ax9 = fig.add_subplot(gs[4, 0])
    ax9.plot(cycles, mean_brightness_p999, 's-', label='P99.9 Mean', linewidth=2, markersize=8, color='C2')
    ax9.plot(cycles, median_brightness_p999, '^--', label='P99.9 Median', linewidth=1.5, markersize=6, color='C2', alpha=0.7)
    # Overlay individual tile P99.9 curves (light green, limited to avoid clutter)
    for i, ratio_p999 in enumerate(all_ratios_p999[:min(50, len(all_ratios_p999))]):
        baseline_p999 = mean_brightness_p999[0] if mean_brightness_p999[0] > 0 else np.max(mean_brightness_p999)
        ax9.plot(cycles, ratio_p999 * baseline_p999, alpha=0.1, linewidth=0.5, color='green')
    ax9.set_xlabel('Cycle', fontsize=11)
    ax9.set_ylabel('P99.9 Brightness', fontsize=11)
    ax9.set_title('Robust Decay Curve (P99.9) - Should be smooth', fontsize=12, fontweight='bold')
    ax9.legend(loc='best')
    ax9.grid(True, alpha=0.3)
    ax9.set_xticks(cycles)
    
    # Plot 10: P99.9 vs Mean ratio comparison (new, bottom right)
    ax10 = fig.add_subplot(gs[4, 1])
    cycle10_ratios_p999 = all_ratios_p999[:, -1]
    cycle10_ratios_mean = all_ratios[:, -1]
    ax10.hist(cycle10_ratios_p999, bins=20, edgecolor='black', alpha=0.7, color='C2', label='P99.9 Ratio')
    ax10.hist(cycle10_ratios_mean, bins=20, edgecolor='black', alpha=0.5, color='C0', label='Mean Ratio')
    ax10.axvline(x=0.2, color='red', linestyle='--', linewidth=2, label='20% threshold')
    ax10.axvline(x=np.mean(cycle10_ratios_p999), color='green', linestyle='-', linewidth=2, 
                 label=f'P99.9 Mean: {np.mean(cycle10_ratios_p999):.1%}')
    ax10.axvline(x=np.mean(cycle10_ratios_mean), color='blue', linestyle='-', linewidth=2, 
                 label=f'Mean: {np.mean(cycle10_ratios_mean):.1%}')
    ax10.set_xlabel('Cycle 10 / Baseline Ratio', fontsize=11)
    ax10.set_ylabel('Number of Tiles', fontsize=11)
    ax10.set_title('Cycle 10 Ratio: Mean vs P99.9 Comparison', fontsize=12, fontweight='bold')
    ax10.legend(loc='best')
    ax10.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(f'Signal Decay Analysis - {run_id} (Aggregated across {n_tiles} tiles)\n'
                 f'Mean (blue) vs P99.9 Percentile (green) - P99.9 should show smoother decay', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Aggregated plot saved to: {output_path}")
    plt.close()


def plot_signal_decay_from_run(run_id, tile_name=None, output_dir=None,
                               cyc_num=SEQ_CYCLE, method='mean', aggregate=True):
    """
    Plot signal decay for a specific run and optionally a specific tile.
    
    Parameters
    ----------
    run_id : str
        Run ID (e.g., '20251128_ZCH_BZ09_Re2_mut_new')
    tile_name : str, optional
        Specific tile name (e.g., 'FocalStack_172.tif'). If None, aggregate all tiles.
    output_dir : str or Path, optional
        Directory to save plots. If None, use readout/tiles/ directory.
    cyc_num : int
        Number of cycles
    method : str
        Aggregation method
    aggregate : bool
        If True and tile_name is None, aggregate all tiles. If False, plot each tile separately.
    """
    dest_directory = os.path.join(BASE_DEST_DIRECTORY, f'{run_id}_processed')
    read_directory = os.path.join(dest_directory, 'readout')
    tiles_output_directory = os.path.join(read_directory, 'tiles')
    
    # Check if tiles directory exists, if not, check for single intensity.csv
    if not os.path.exists(tiles_output_directory):
        single_intensity_file = Path(read_directory) / 'intensity.csv'
        if single_intensity_file.exists():
            print(f"Note: Tiles directory not found, but found single intensity.csv file.")
            print(f"      Using --intensity_file mode instead.")
            if output_dir is None:
                output_dir = read_directory
            else:
                os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, 'intensity_decay.png')
            plot_signal_decay_from_file(str(single_intensity_file), output_path, cyc_num, method)
            return
        else:
            raise FileNotFoundError(
                f"Tiles output directory not found: {tiles_output_directory}\n"
                f"  Also checked for single intensity.csv in: {read_directory}\n"
                f"  Neither location contains intensity data."
            )
    
    if output_dir is None:
        output_dir = tiles_output_directory
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    # Find intensity files
    if tile_name:
        tile_stem = Path(tile_name).stem
        # Try multiple naming patterns
        intensity_file = os.path.join(tiles_output_directory, f'{tile_stem}_intensity_filtered.csv')
        if not os.path.exists(intensity_file):
            intensity_file = os.path.join(tiles_output_directory, f'{tile_stem}_intensity_raw.csv')
        if not os.path.exists(intensity_file):
            raise FileNotFoundError(
                f"Intensity file not found for tile {tile_name}.\n"
                f"  Searched for:\n"
                f"    - {tile_stem}_intensity_filtered.csv\n"
                f"    - {tile_stem}_intensity_raw.csv\n"
                f"  In directory: {tiles_output_directory}"
            )
        
        output_path = os.path.join(output_dir, f'{tile_stem}_signal_decay.png')
        plot_signal_decay_from_file(intensity_file, output_path, cyc_num, method)
    else:
        if aggregate:
            # Aggregate all tiles
            aggregate_all_tiles(run_id, output_dir, cyc_num, method)
        else:
            # Process all tiles separately
            intensity_files = sorted(Path(tiles_output_directory).glob('*_intensity_filtered.csv'))
            if len(intensity_files) == 0:
                intensity_files = sorted(Path(tiles_output_directory).glob('*_intensity_raw.csv'))
            if len(intensity_files) == 0:
                raise ValueError(f"No intensity files found in {tiles_output_directory}")
            
            print(f"Found {len(intensity_files)} intensity files. Processing...")
            for intensity_file in intensity_files:
                tile_stem = intensity_file.stem.replace('_intensity_filtered', '')
                output_path = os.path.join(output_dir, f'{tile_stem}_signal_decay.png')
                try:
                    plot_signal_decay_from_file(intensity_file, output_path, cyc_num, method)
                except Exception as e:
                    print(f"Error processing {intensity_file}: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot signal decay QC plots')
    parser.add_argument('--run_id', type=str, required=True,
                       help='Run ID (e.g., 20251128_ZCH_BZ09_Re2_mut_new)')
    parser.add_argument('--tile', type=str, default=None,
                       help='Specific tile name (e.g., FocalStack_172.tif). If not specified, process all tiles.')
    parser.add_argument('--intensity_file', type=str, default=None,
                       help='Direct path to intensity CSV file (alternative to run_id)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for plots')
    parser.add_argument('--method', type=str, default='mean',
                       choices=['mean', 'median', 'sum', 'total_mean'],
                       help='Aggregation method for calculating brightness')
    parser.add_argument('--cycles', type=int, default=SEQ_CYCLE,
                       help='Number of cycles')
    parser.add_argument('--aggregate', action='store_true', default=True,
                       help='Aggregate all tiles (default: True). Use --no-aggregate to plot each tile separately.')
    parser.add_argument('--no-aggregate', dest='aggregate', action='store_false',
                       help='Plot each tile separately instead of aggregating')
    
    args = parser.parse_args()
    
    if args.intensity_file:
        # Direct file mode
        plot_signal_decay_from_file(
            args.intensity_file,
            output_path=args.output_dir,
            cyc_num=args.cycles,
            method=args.method
        )
    else:
        # Run ID mode
        plot_signal_decay_from_run(
            args.run_id,
            tile_name=args.tile,
            output_dir=args.output_dir,
            cyc_num=args.cycles,
            method=args.method,
            aggregate=args.aggregate
        )


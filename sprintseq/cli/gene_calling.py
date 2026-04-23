import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sprintseq.gene_calling import correct_intensity, map_genes


def plot_mapping_qc(mapping_result, output_path, method_name='postcode',
                    prob_high=0.9, prob_mid=0.8):
    """Plot QC summary for a mapping result into a single composite figure.

    Combines the diagnostic plots from inspect_readout.ipynb:
    - Probability distribution (all spots)
    - Entropy distribution (all spots)
    - Probability vs Entropy 2D density
    - Entropy distribution for high-probability subset (Prob > prob_mid)
    - Cumulative pass rate vs probability threshold
    - Top mapped gene counts (high-confidence subset, Prob > prob_high)

    Parameters
    ----------
    mapping_result : pd.DataFrame
        DataFrame returned by map_genes (must contain 'Gene', 'Probability',
        and 'Entropy' columns for postcode output).
    output_path : str or Path
        Where to save the resulting PNG figure.
    method_name : str
        Method name (used in figure title).
    prob_high : float
        High-confidence probability threshold.
    prob_mid : float
        Medium-confidence probability threshold (used for entropy subset).
    """
    import matplotlib.pyplot as plt

    required_cols = {'Probability', 'Entropy', 'Gene'}
    missing = required_cols - set(mapping_result.columns)
    if missing:
        print(f"    ! Skipping QC plot, missing columns: {missing}")
        return

    prob = mapping_result['Probability'].to_numpy()
    entropy = mapping_result['Entropy'].to_numpy()
    total = len(mapping_result)
    n_high = int((prob > prob_high).sum())
    n_mid = int((prob > prob_mid).sum())

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f'Mapping QC ({method_name}) - {total:,} spots  |  '
        f'Prob>{prob_high}: {n_high:,} ({n_high / total * 100:.1f}%)  |  '
        f'Prob>{prob_mid}: {n_mid:,} ({n_mid / total * 100:.1f}%)',
        fontsize=14,
    )

    # (1) Probability histogram
    ax = axes[0, 0]
    ax.hist(prob, bins=100, color='steelblue', edgecolor='none')
    ax.axvline(prob_high, color='red', linestyle='--', alpha=0.7,
               label=f'Prob = {prob_high}')
    ax.axvline(prob_mid, color='orange', linestyle='--', alpha=0.7,
               label=f'Prob = {prob_mid}')
    ax.set_xlabel('Probability')
    ax.set_ylabel('Count')
    ax.set_title('Probability distribution (all spots)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (2) Entropy histogram
    ax = axes[0, 1]
    ax.hist(entropy, bins=100, color='seagreen', edgecolor='none')
    ax.set_xlabel('Entropy')
    ax.set_ylabel('Count')
    ax.set_title('Entropy distribution (all spots)')
    ax.grid(True, alpha=0.3)

    # (3) 2D density: Probability vs Entropy
    ax = axes[0, 2]
    h = ax.hist2d(prob, entropy, bins=100, cmap='viridis',
                  cmin=1)
    ax.set_xlabel('Probability')
    ax.set_ylabel('Entropy')
    ax.set_title('Probability vs Entropy')
    fig.colorbar(h[3], ax=ax, label='Count')

    # (4) Entropy distribution for Probability > prob_mid
    ax = axes[1, 0]
    mask_mid = prob > prob_mid
    if mask_mid.any():
        ax.hist(entropy[mask_mid], bins=100, color='darkorange',
                edgecolor='none')
    ax.set_xlabel('Entropy')
    ax.set_ylabel('Count')
    ax.set_title(f'Entropy distribution (Prob > {prob_mid})')
    ax.grid(True, alpha=0.3)

    # (5) Cumulative pass rate vs probability threshold
    ax = axes[1, 1]
    thresholds = np.linspace(0.0, 1.0, 101)
    pass_rate = [(prob > t).mean() * 100 for t in thresholds]
    ax.plot(thresholds, pass_rate, color='purple', linewidth=2)
    ax.axvline(prob_high, color='red', linestyle='--', alpha=0.7,
               label=f'Prob > {prob_high}: {n_high / total * 100:.1f}%')
    ax.axvline(prob_mid, color='orange', linestyle='--', alpha=0.7,
               label=f'Prob > {prob_mid}: {n_mid / total * 100:.1f}%')
    ax.set_xlabel('Probability threshold')
    ax.set_ylabel('Spots passing (%)')
    ax.set_title('Cumulative pass rate vs threshold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (6) Top genes (high-confidence subset)
    ax = axes[1, 2]
    high_subset = mapping_result.loc[mask_mid & mapping_result['Gene'].notna(),
                                     'Gene']
    if len(high_subset) > 0:
        top_genes = high_subset.value_counts().head(20)
        ax.barh(range(len(top_genes)), top_genes.values[::-1],
                color='teal')
        ax.set_yticks(range(len(top_genes)))
        ax.set_yticklabels(top_genes.index[::-1], fontsize=8)
        ax.set_xlabel('Count')
        ax.set_title(f'Top 20 genes (Prob > {prob_mid})')
    else:
        ax.text(0.5, 0.5, 'No high-confidence spots',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'Top genes (Prob > {prob_mid})')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

# basic paths
BASE_DIR = Path(r'\\10.10.10.1\NAS Processed Images')

# Configuration
DEFAULT_RUN_ID = 'example_dataset'  # Default Run ID to process

# Reference file configuration (default path, can be overridden via --ref-file)
REF_FOLDER = Path(__file__).parent.parent / 'reference'  # Reference folder path
DEFAULT_REF_FILE = REF_FOLDER / 'ZCH_TNBC_marker_genes.csv'  # Default reference file path

# sequence calling
SEQ_CYCLE = 10  
CHANNELS = ['cy3', 'cy5']


def test_single_tile_mapping(run_id, tile_name, ref_file, method='postcode', **kwargs):
    """Test function to run gene mapping on a single tile's intensity data.
    
    This function reads a specific tile's raw intensity file, and runs the specified
    mapping method on it. Useful for quick testing.
    
    Parameters
    ----------
    run_id : str
        Run ID (e.g., '20251211_ZCH_BZ29_TNBC_marker_test1')
    tile_name : str
        Tile name (e.g., 'FocalStack_172.tif')
    ref_file : str or Path
        Path to reference codebook file
    method : str
        Mapping method to use (default: 'postcode')
    **kwargs : dict
        Additional arguments passed to map_genes
    """
    print("=" * 80)
    print("Single Tile Mapping Test Mode")
    print("=" * 80)
    print(f"Run ID: {run_id}")
    print(f"Tile: {tile_name}")
    print(f"Method: {method}")
    print("=" * 80)
    
    # Construct paths
    base_dir = Path(r'\\10.10.10.1\NAS Processed Images')
    dest_dir = base_dir / f'{run_id}_processed'
    read_dir = dest_dir / 'readout'
    tiles_output_dir = read_dir / 'tiles'
    
    # Construct tile intensity filename
    # Format: {tile_stem}_intensity.csv
    tile_stem = Path(tile_name).stem
    intensity_file = tiles_output_dir / f'{tile_stem}_intensity.csv'
    
    if not intensity_file.exists():
        raise FileNotFoundError(f"Intensity file not found: {intensity_file}")
        
    print(f"Loading intensity from: {intensity_file}")
    
    # Read intensity dataframe
    intensity_df = pd.read_csv(intensity_file)
    
    # Ensure index column exists
    if 'index' not in intensity_df.columns:
        intensity_df.insert(0, 'index', range(len(intensity_df)))
        
    print(f"Loaded {len(intensity_df)} spots")
    print()
    
    # Run mapping
    print(f"Running {method} mapping...")
    
    # Request diagnostics if method is postcode
    if method == 'postcode':
        kwargs['return_diagnostics'] = True
        
    try:
        # Call map_genes
        mapping_output = map_genes(
            intensity_df,
            str(ref_file),
            method=method,
            verbose=True,
            **kwargs
        )
        
        # Unpack result if diagnostics returned
        diagnostics = {}
        if isinstance(mapping_output, tuple):
            result_df, diagnostics = mapping_output
        else:
            result_df = mapping_output
            
        # Save result
        output_file = read_dir / f'{tile_stem}_mapping_{method}.csv'
        result_df.to_csv(output_file, index=False)
        
        print()
        print("=" * 80)
        print("Test completed successfully!")
        print(f"Results saved to: {output_file}")
        
        # Plot diagnostics (Convergence Curve)
        if method == 'postcode' and 'losses' in diagnostics:
            losses = diagnostics['losses']
            if len(losses) > 0:
                print(f"Generating convergence plot (ELBO loss) for {len(losses)} iterations...")
                try:
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(10, 6))
                    plt.plot(losses, marker='o', linestyle='-', markersize=3, alpha=0.7)
                    plt.title(f'PoSTcode Convergence (ELBO Loss) - {tile_stem}')
                    plt.xlabel('Iteration')
                    plt.ylabel('Loss (ELBO)')
                    plt.grid(True, alpha=0.3)
                    
                    # Save plot
                    plot_file = read_dir / f'{tile_stem}_convergence_{method}.png'
                    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"Convergence plot saved to: {plot_file}")
                except Exception as e:
                    print(f"Could not generate plot: {e}")
        
        # Print summary stats
        mapped_count = result_df['Gene'].notna().sum()
        print(f"Mapping Rate: {mapped_count}/{len(result_df)} ({mapped_count/len(result_df)*100:.1f}%)")
        
        if 'Probability' in result_df.columns:
            print(f"Avg Probability: {result_df['Probability'].mean():.3f}")
            
    except Exception as e:
        print(f"Error during mapping: {e}")
        import traceback
        traceback.print_exc()

def run_pipeline(run_id=None, ref_file=None, seq_cycle=None, channels=None):
    """Main function to run intensity correction + gene mapping for a RUN_ID.

    Parameters
    ----------
    run_id : str
        Run ID (e.g., '20251128_ZCH_BZ09_Re2_mut_new').
    ref_file : str or Path
        Path to reference codebook file (CSV with Barcode, Gene columns).
    seq_cycle : int, optional
        Number of sequencing cycles to decode. Defaults to SEQ_CYCLE (=10).
    channels : list[str], optional
        SBS spot channels. Defaults to CHANNELS (=['cy3','cy5']).
    """
    # Validate required parameters
    if run_id is None:
        raise ValueError("run_id is required and cannot be None")

    if ref_file is None:
        raise ValueError("ref_file is required and cannot be None")

    ref_file = Path(ref_file)
    seq_cycle = SEQ_CYCLE if seq_cycle is None else seq_cycle
    channels = list(CHANNELS) if channels is None else list(channels)
    
    dest_dir = BASE_DIR / f'{run_id}_processed'
    read_dir = dest_dir / 'readout'

    # Load complete intensity produced by readout (position.csv + intensity.csv, already deduplicated)
    position_output = read_dir / 'position.csv'
    intensity_output = read_dir / 'intensity.csv'
    if not position_output.exists() or not intensity_output.exists():
        raise FileNotFoundError(
            f"Readout output not found. Run readout first to generate {position_output} and {intensity_output}"
        )

    print("=" * 80)
    print("Gene calling (load complete intensity from readout)")
    print("=" * 80)
    print(f"Run ID: {run_id}")
    print(f"Readout directory: {read_dir}")
    print()

    print("Step 1: Loading position and intensity from readout...")
    position_df = pd.read_csv(position_output)
    intensity_df = pd.read_csv(intensity_output)
    global_intensity_df = position_df[['index', 'Y', 'X']].merge(
        intensity_df, on='index', how='inner'
    )
    print(f"  Loaded {len(global_intensity_df)} points (position + intensity)")
    print()

    # Define mapping methods to run
    # Moved here to decide if intensity correction is needed
    mapping_methods = [
        {
            'name': 'postcode',
            'method': 'postcode',
            'kwargs': {
                'cyc_num': seq_cycle,
                'channels': channels,
                'batch_size': 100000,
                'num_iter': 60,
                'verbose': True,
                'return_diagnostics': True
            }
        },
        # Uncomment to run threshold method
        # {
        #     'name': 'threshold',
        #     'method': 'threshold',
        #     'kwargs': {
        #         'thresholds': {'cy3': 50, 'cy5': 50},
        #         'adaptive_threshold': False,  # Use fixed thresholds
        #         'cyc_num': seq_cycle,
        #         'channels': channels,
        #         'verbose': True
        #     }
        # }
    ]

    # Check if any method needs corrected intensity
    need_intensity_correction = any(m['method'] != 'postcode' for m in mapping_methods)

    # Step 2: Global intensity correction
    correction_info_path = read_dir / 'global_correction_info.json'
    intensity_corrected_output = read_dir / 'intensity_corrected.csv'
    
    if need_intensity_correction:
        print("Step 2: Applying global intensity correction...")
        print("-" * 80)
        
        # Validate ref_file exists (required for intensity correction)
        if not ref_file.exists():
            raise FileNotFoundError(f"Reference file not found at {ref_file}. Required for intensity correction.")
        
        # Apply global correction using all data
        # This includes: channel balance correction, decay correction, and phasing correction
        intensity_df_corrected, correction_info = correct_intensity(
            global_intensity_df,
            cyc_num=seq_cycle,
            channels=channels,
            estimate_phasing=True,  # Automatically estimate phasing rates using all data
            phasing_estimation_method='grid_search',
            ref_file=str(ref_file),
            correct_channel_balance=True,  # Enable channel balance correction
            channel_balance_method='codebook',  # Use codebook method (ref_file is required)
            verbose=True
        )
        
        # Merge corrected intensities back to original dataframe and convert to uint16
        intensity_cols_corrected = [col for col in intensity_df_corrected.columns if col.startswith('cyc_')]
        for col in intensity_cols_corrected:
            if col in intensity_df_corrected.columns:
                # Round and convert to uint16 to save memory
                global_intensity_df[col] = np.round(intensity_df_corrected[col].values).clip(0, 65535).astype(np.uint16)
        
        # Free memory: delete corrected dataframe after copying values
        del intensity_df_corrected
        import gc
        gc.collect()
        
        # Save correction info
        import json
        info_to_save = {}
        for key, value in correction_info.items():
            if isinstance(value, np.ndarray):
                info_to_save[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                info_to_save[key] = float(value)
            else:
                info_to_save[key] = value
        
        with open(correction_info_path, 'w') as f:
            json.dump(info_to_save, f, indent=2)
        
        # Print correction summary
        if correction_info.get('balance_corrected', False):
            balance_factors = correction_info.get('balance_factors', {})
            print(f"  Channel balance applied: cy3={balance_factors.get('cy3', 1.0):.3f}, "
                  f"cy5={balance_factors.get('cy5', 1.0):.3f}")
        
        print(f"  Phasing correction: phasing_rate={correction_info.get('phasing_rate', 'N/A')}, "
                f"prephasing_rate={correction_info.get('prephasing_rate', 'N/A')}")
        print(f"  Correction info saved to: {correction_info_path}")
        
        # Save corrected intensity data (with index)
        print()
        print("  Saving corrected intensity data...")
        if intensity_corrected_output.exists():
            print(f"  File already exists, skipping save: {intensity_corrected_output}")
        else:
            intensity_corrected_df = global_intensity_df[[col for col in global_intensity_df.columns if col.startswith('cyc_')]].copy()
            intensity_corrected_df.insert(0, 'index', global_intensity_df['index'].values)
            intensity_corrected_df.to_csv(intensity_corrected_output, index=False)
            print(f"  Saved to: {intensity_corrected_output}")
        print()
        
    else:
        print("Step 2: Skipping global intensity correction (not required for selected methods)...")
        correction_info = {'skipped': True}
    
    # Step 3: Gene mapping using multiple methods
    print("Step 3: Gene mapping (testing multiple methods)...")
    print("-" * 80)
    
    if not ref_file.exists():
        raise FileNotFoundError(f"Reference file not found at {ref_file}")
    
    # Get balance factors from correction info (if available)
    # Data was already balanced in Step 2, so we can use those factors for per_round_max
    balance_factors = None
    if correction_info.get('balance_corrected', False):
        balance_factors = correction_info.get('balance_factors', None)
    
    # Test each mapping method and save immediately after each completes
    mapping_stats = {}
    saved_files = []
    
    for method_config in mapping_methods:
        base_method_name = method_config['name']
        method = method_config['method']
        kwargs = method_config['kwargs']

        # For intensity-based mapping, encode similarity_metric into the method name
        # so that output files follow mapping_intensity_{similarity_metric}.csv
        if base_method_name == 'intensity_direct':
            sim_metric = kwargs.get('similarity_metric', 'metric_distance')
            method_name = f"intensity_{sim_metric}"
        else:
            method_name = base_method_name
        
        print(f"\n  Testing method: {method_name} ({method})")
        print("  " + "-" * 76)
        
        try:
            mapping_output = map_genes(
                global_intensity_df,
                str(ref_file),
                method=method,
                **kwargs
            )
            
            # Unpack result if diagnostics returned
            diagnostics = {}
            if isinstance(mapping_output, tuple):
                mapping_result, diagnostics = mapping_output
            else:
                mapping_result = mapping_output
            
            # Calculate statistics BEFORE saving (need to access DataFrame)
            total_points = len(mapping_result)
            mapped_count = mapping_result['Gene'].notna().sum() if 'Gene' in mapping_result.columns else 0
            mapping_rate = (mapped_count / total_points * 100) if total_points > 0 else 0
            
            # Get additional metrics if available
            if 'similarity' in mapping_result.columns:
                avg_similarity = mapping_result['similarity'].mean()
                mapping_stats[method_name] = {
                    'total_points': total_points,
                    'mapped_count': mapped_count,
                    'mapping_rate': mapping_rate,
                    'avg_similarity': avg_similarity
                }
            elif 'confidence' in mapping_result.columns:
                avg_confidence = mapping_result['confidence'].mean()
                mapping_stats[method_name] = {
                    'total_points': total_points,
                    'mapped_count': mapped_count,
                    'mapping_rate': mapping_rate,
                    'avg_confidence': avg_confidence
                }
            else:
                mapping_stats[method_name] = {
                    'total_points': total_points,
                    'mapped_count': mapped_count,
                    'mapping_rate': mapping_rate
                }
            
            print(f"  ✓ Completed: {mapped_count}/{total_points} mapped ({mapping_rate:.1f}%)")
            
            # Immediately save the result and free memory
            print(f"  Saving {method_name} result...")
            mapping_output_path = read_dir / f'mapping_{method_name}.csv'
            try:
                mapping_result.to_csv(mapping_output_path, index=False)
                saved_files.append(mapping_output_path)
                print(f"    ✓ Saved to: {mapping_output_path}")
                
                # Plot diagnostics (Convergence Curve) for PoSTcode
                if method == 'postcode' and 'losses' in diagnostics:
                    losses = diagnostics['losses']
                    if len(losses) > 0:
                        print(f"    Generating convergence plot ({len(losses)} iter)...")
                        try:
                            import matplotlib.pyplot as plt
                            plt.figure(figsize=(10, 6))
                            plt.plot(losses, marker='o', linestyle='-', markersize=3, alpha=0.7)
                            plt.title(f'PoSTcode Convergence (ELBO Loss) - Full Run')
                            plt.xlabel('Iteration')
                            plt.ylabel('Loss (ELBO)')
                            plt.grid(True, alpha=0.3)
                            
                            # Save plot
                            plot_file = read_dir / f'convergence_{method_name}.png'
                            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                            plt.close()
                            print(f"    ✓ Plot saved to: {plot_file}")
                        except Exception as e:
                            print(f"    ! Could not generate plot: {e}")

                # QC plot for postcode (composite figure with probability/entropy diagnostics)
                if method == 'postcode':
                    qc_plot_file = read_dir / f'mapping_qc_{method_name}.png'
                    print(f"    Generating QC summary figure...")
                    try:
                        plot_mapping_qc(mapping_result, qc_plot_file,
                                        method_name=method_name)
                        print(f"    ✓ QC plot saved to: {qc_plot_file}")
                    except Exception as e:
                        print(f"    ! Could not generate QC plot: {e}")
                        import traceback
                        traceback.print_exc()

                if 'avg_similarity' in mapping_stats[method_name]:
                    print(f"    Avg similarity: {mapping_stats[method_name]['avg_similarity']:.3f}")
                elif 'avg_confidence' in mapping_stats[method_name]:
                    print(f"    Avg confidence: {mapping_stats[method_name]['avg_confidence']:.3f}")
            except Exception as e:
                print(f"    ✗ Failed to save: {str(e)}")
                import traceback
                traceback.print_exc()
            
            # Free memory: delete the large DataFrame immediately after saving
            del mapping_result
            import gc
            gc.collect()
            
        except Exception as e:
            print(f"  ✗ Failed: {str(e)}")
            import traceback
            traceback.print_exc()
            mapping_stats[method_name] = {
                'error': str(e)
            }
            # Continue to next method even if this one failed
    
    print()
    
    # Summary statistics
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Run ID: {run_id}")
    print(f"Total points: {len(position_df)}")
    print()
    print("Output files:")
    print(f"  - position.csv: {position_output}")
    print(f"  - intensity.csv: {intensity_output}")
    print(f"  - intensity_corrected.csv: {intensity_corrected_output}")
    print(f"  - global_correction_info.json: {correction_info_path}")
    print()
    print("Mapping results:")
    for method_name, stats in mapping_stats.items():
        if 'error' not in stats:
            print(f"  - mapping_{method_name}.csv: {stats.get('mapped_count', 0)}/{stats.get('total_points', 0)} "
                  f"({stats.get('mapping_rate', 0):.1f}%)")
            if 'avg_similarity' in stats:
                print(f"    (Avg similarity: {stats['avg_similarity']:.3f})")
            elif 'avg_confidence' in stats:
                print(f"    (Avg confidence: {stats['avg_confidence']:.3f})")
        else:
            print(f"  - mapping_{method_name}.csv: Failed")
    print()
    print("Mapping method comparison:")
    for method_name, stats in mapping_stats.items():
        if 'error' not in stats:
            print(f"  {method_name:20s}: {stats.get('mapping_rate', 0):6.1f}% "
                  f"({stats.get('mapped_count', 0):,} points)")
    print("=" * 80)


def main():
    """`python -m sprintseq.cli.gene_calling` fallback entry point. The canonical CLI is `sprintseq gene-calling` (see sprintseq.cli.main)."""
    from sprintseq.cli import parse_channels
    parser = argparse.ArgumentParser(description='Gene calling pipeline for SPRINTseq')
    parser.add_argument('--run-id', type=str, required=True,
                        help='Run ID to process (e.g., "20251211_ZCH_BZ29_TNBC_marker_test1")')
    parser.add_argument('--ref-file', type=str, required=True,
                        help='Path to reference codebook file (e.g., CSV file with gene barcodes)')
    parser.add_argument('--seq-cycles', type=int, default=None,
                        help=f'Number of sequencing cycles to decode. Default: {SEQ_CYCLE}')
    parser.add_argument('--channels', type=parse_channels, default=None,
                        help=f'Comma-separated SBS channels. Default: {",".join(CHANNELS)}')
    args = parser.parse_args()
    run_pipeline(run_id=args.run_id, ref_file=args.ref_file,
                 seq_cycle=args.seq_cycles, channels=args.channels)


if __name__ == "__main__":
    main()


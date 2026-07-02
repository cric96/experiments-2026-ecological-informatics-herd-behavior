import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def compare_and_plot():
    # Set context and style for premium aesthetics
    sns.set_theme(style="whitegrid")
    sns.set_context("talk", font_scale=0.9)  # Large, readable labels
    
    # 1. Load Real Data
    print("Loading real dataset...")
    real_dataset = pd.read_csv("data/velocity_reconstruction.csv", delimiter=' ', comment='#', names=list(range(10)))
    real_dataset.rename(columns={0: 'time'}, inplace=True)
    real_dataset = real_dataset.melt(id_vars=['time'], var_name='node', value_name='velocity')
    real_dataset['velocity'] = real_dataset['velocity'] / 3.6  # convert to m/s
    real_dataset = real_dataset.dropna()
    real_dataset['source'] = 'Real (KABR Dataset)'
    
    # 2. Load Simulated Data
    print("Loading simulated dataset...")
    means = pickle.load(open('data_summary_mean', 'rb'))
    v_sim = means['velocity_simulation']
    sim_df = v_sim.to_dataframe().reset_index()
    sim_df = sim_df.melt(
        id_vars=['time', 'intrinsicForwardCoefficient', 'intrinsicLateralMultiplier', 'NumberOfHerds'],
        value_vars=[col for col in sim_df.columns if col.startswith('node-')],
        var_name='node', value_name='velocity'
    )
    sim_df['velocity'] = sim_df['velocity'] / 3.6  # convert to m/s
    sim_df = sim_df.dropna()
    
    # Define interesting parameters to compare
    # Let's find some representative parameters
    lateral_vals = sorted(sim_df['intrinsicLateralMultiplier'].unique())
    forward_vals = sorted(sim_df['intrinsicForwardCoefficient'].unique())
    
    # Pick a few representative values for forward coeff (low, mid, high)
    selected_forwards = [forward_vals[0], forward_vals[len(forward_vals)//2], forward_vals[-1]]
    selected_lateral = lateral_vals[len(lateral_vals)//2] # typically 0.5
    
    # Filter simulation to representative cases
    sim_rep_df = sim_df[
        (sim_df['intrinsicLateralMultiplier'] == selected_lateral) & 
        (sim_df['intrinsicForwardCoefficient'].isin(selected_forwards))
    ].copy()
    
    # Add descriptive label to simulation cases
    sim_rep_df['source'] = sim_rep_df.apply(
        lambda r: f"Sim (Fwd={r['intrinsicForwardCoefficient']:.2f}, Lat={r['intrinsicLateralMultiplier']:.2f})", 
        axis=1
    )
    
    # Combine datasets for unified plots where needed
    combined_df = pd.concat([
        real_dataset[['velocity', 'source']], 
        sim_rep_df[['velocity', 'source']]
    ], ignore_index=True)
    
    # Create output dir
    os.makedirs('charts', exist_ok=True)
    
    # Setup Figure with 3 Panels
    fig, axes = plt.subplots(1, 3, figsize=(22, 6.5), gridspec_kw={'width_ratios': [1.1, 1, 0.9]})
    fig.suptitle("Comparison of Real vs. Simulated Velocities Distribution", fontsize=18, fontweight='bold', y=1.02)
    
    palette_colors = ['#1f77b4', '#2ca02c', '#9467bd', '#d62728'] # Real (Blue), Sim 1 (Green), Sim 2 (Purple), Sim 3 (Red)
    sources = [real_dataset['source'].iloc[0]] + sorted(sim_rep_df['source'].unique())
    color_map = dict(zip(sources, palette_colors))
    
    # --- PANEL A: Probability Density Function (KDE) ---
    ax = axes[0]
    # We use density/relative frequency for comparison so shape is visible despite sample size difference
    sns.kdeplot(
        data=real_dataset, x='velocity', ax=ax, label=real_dataset['source'].iloc[0],
        color=color_map[real_dataset['source'].iloc[0]], linewidth=3, fill=True, alpha=0.15, bw_adjust=1.5
    )
    for src in sorted(sim_rep_df['source'].unique()):
        sns.kdeplot(
            data=sim_rep_df[sim_rep_df['source'] == src], x='velocity', ax=ax, label=src,
            color=color_map[src], linewidth=2, bw_adjust=1.5
        )
    ax.set_title("A. Velocity Probability Density (KDE)", fontsize=14, fontweight='semibold', pad=12)
    ax.set_xlabel("Velocity (m/s)")
    ax.set_ylabel("Probability Density")
    ax.set_xlim(0, 7)
    ax.legend(fontsize=10, loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    
    # --- PANEL B: Empirical Cumulative Distribution Function (ECDF) ---
    ax = axes[1]
    # ECDF is the cleanest way to compare shapes of distributions without any binning/bandwidth assumptions
    sns.ecdfplot(
        data=real_dataset, x='velocity', ax=ax, label=real_dataset['source'].iloc[0],
        color=color_map[real_dataset['source'].iloc[0]], linewidth=3
    )
    for src in sorted(sim_rep_df['source'].unique()):
        sns.ecdfplot(
            data=sim_rep_df[sim_rep_df['source'] == src], x='velocity', ax=ax, label=src,
            color=color_map[src], linewidth=2
        )
    ax.set_title("B. Empirical Cumulative Distribution (ECDF)", fontsize=14, fontweight='semibold', pad=12)
    ax.set_xlabel("Velocity (m/s)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 1.05)
    
    # --- PANEL C: Box & Violin Plot (Quantile Comparison) ---
    ax = axes[2]
    combined_df['display_source'] = combined_df['source'].apply(
        lambda x: x.replace(' (KABR Dataset)', '').replace('Sim (', '').replace(')', '')
    )
    display_color_map = {
        k.replace(' (KABR Dataset)', '').replace('Sim (', '').replace(')', ''): v 
        for k, v in color_map.items()
    }
    
    sns.violinplot(
        data=combined_df, y='display_source', x='velocity', ax=ax, hue='display_source',
        palette=display_color_map, orient='h', inner='quart', linewidth=1.5, legend=False
    )
    ax.set_title("C. Quartiles & Range (Violin Plot)", fontsize=14, fontweight='semibold', pad=12)
    ax.set_xlabel("Velocity (m/s)")
    ax.set_ylabel("")
    ax.set_xlim(0, 7)

    
    plt.tight_layout()
    
    # Save as PNG (high quality)
    output_png = 'charts/velocity_comparison.png'
    fig.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_png}")
    plt.close(fig)
    
    # 3. Print Descriptive Statistics
    print("\n" + "="*50)
    print("DESCRIPTIVE STATISTICS COMPARISON")
    print("="*50)
    
    stats_data = []
    # Real Stats
    real_desc = real_dataset['velocity'].describe(percentiles=[0.25, 0.5, 0.75, 0.9])
    stats_data.append({
        'Source': 'Real (KABR)',
        'Mean (m/s)': real_desc['mean'],
        'SD (m/s)': real_desc['std'],
        'Min (m/s)': real_desc['min'],
        '25% (m/s)': real_desc['25%'],
        'Median (m/s)': real_desc['50%'],
        '75% (m/s)': real_desc['75%'],
        '90% (m/s)': real_desc['90%'],
        'Max (m/s)': real_desc['max']
    })
    
    # Sim Stats
    for src in sorted(sim_rep_df['source'].unique()):
        subset = sim_rep_df[sim_rep_df['source'] == src]['velocity']
        desc = subset.describe(percentiles=[0.25, 0.5, 0.75, 0.9])
        stats_data.append({
            'Source': src.replace('Sim (', '').replace(')', ''),
            'Mean (m/s)': desc['mean'],
            'SD (m/s)': desc['std'],
            'Min (m/s)': desc['min'],
            '25% (m/s)': desc['25%'],
            'Median (m/s)': desc['50%'],
            '75% (m/s)': desc['75%'],
            '90% (m/s)': desc['90%'],
            'Max (m/s)': desc['max']
        })
        
    stats_df = pd.DataFrame(stats_data)
    print(stats_df.to_string(index=False, index_names=False, justify='center', float_format=lambda x: f"{x:.3f}"))
    
    # Save stats to markdown file for easy viewing
    with open('charts/statistics_comparison.md', 'w') as f:
        f.write("# Comparison Statistics Table\n\n")
        f.write(stats_df.to_markdown(index=False, floatfmt=".3f"))
        f.write("\n")

if __name__ == "__main__":
    compare_and_plot()

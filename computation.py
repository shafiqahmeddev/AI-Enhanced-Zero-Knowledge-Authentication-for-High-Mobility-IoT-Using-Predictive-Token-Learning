import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set modern style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create directory to save the images
output_dir = "."
os.makedirs(output_dir, exist_ok=True)

# Define protocols and authentication phases
protocols = ['ZKPAS', '[19]', '[8]', '[15]', '[10]']
phases = ['Initial\nHandshake', 'Sliding Window\nLink', 'Continuous\nValidation', 'Cross-Domain\nTransfer', 'Predictive\nPre-computation']

# Define computational overhead values (ms) from the text
computational_data = {
    'ZKPAS': [1.89, 0.127, 0.094, 3.24, 2.15],
    '[19]': [9.43, 3.78, 2.85, 12.67, 8.91],
    '[8]': [4.69, 1.62, 1.33, 7.42, 5.28],
    '[15]': [6.84, 2.47, 1.98, 9.15, 6.73],
    '[10]': [7.21, 2.91, 2.16, 10.38, 7.89]
}

# Create figure with enhanced styling
fig, ax = plt.subplots(figsize=(16, 10))
fig.patch.set_facecolor('#f8f9fa')

# Enhanced color palette
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
patterns = ['', '///', '...', 'xxx', '|||']

x = np.arange(len(phases))
width = 0.15

# Draw enhanced bar plot
bars = []
for i, protocol in enumerate(protocols):
    offset = (i - 2) * width
    values = computational_data[protocol]
    
    # Create bars with gradient effect
    bar = ax.bar(x + offset, values, width, 
                label=protocol, 
                color=colors[i], 
                alpha=0.8,
                edgecolor='white',
                linewidth=1.5,
                hatch=patterns[i] if i > 0 else None)
    
    # Add value labels on top of bars
    for j, v in enumerate(values):
        ax.text(x[j] + offset, v + 0.2, f'{v}', 
               ha='center', va='bottom', fontweight='bold', 
               fontsize=9, color='#2c3e50')
    
    bars.append(bar)

# Enhanced styling
ax.set_xlabel('Authentication Phase', fontsize=14, fontweight='bold', color='#2c3e50')
ax.set_ylabel('Computational Overhead (ms)', fontsize=14, fontweight='bold', color='#2c3e50')
ax.set_title('Computational Overhead Across Authentication Phases\nZero-Knowledge Predictive Authentication System (ZKPAS)', 
             fontsize=16, fontweight='bold', color='#2c3e50', pad=20)

# Enhanced x-axis
ax.set_xticks(x)
ax.set_xticklabels(phases, fontsize=11, fontweight='bold', color='#34495e')
ax.tick_params(axis='x', pad=10)

# Enhanced y-axis
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.tick_params(axis='y', labelsize=11, color='#34495e')

# Enhanced legend
legend = ax.legend(loc='upper right', frameon=True, shadow=True, 
                  fancybox=True, framealpha=0.9, fontsize=11)
legend.get_frame().set_facecolor('#ffffff')
legend.get_frame().set_edgecolor('#bdc3c7')

# Enhanced grid
ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.3, color='#7f8c8d')
ax.set_axisbelow(True)

# Add background gradient effect
ax.set_facecolor('#fdfdfd')

# Add a subtle border
for spine in ax.spines.values():
    spine.set_edgecolor('#bdc3c7')
    spine.set_linewidth(1.5)

# Add performance indicator
max_value = max([max(values) for values in computational_data.values()])
ax.axhline(y=max_value*0.1, color='#27ae60', linestyle=':', linewidth=2, alpha=0.7)
ax.text(len(phases)-1, max_value*0.12, 'Optimal Range', 
        ha='right', va='bottom', fontsize=10, color='#27ae60', fontweight='bold')

# Add subtitle
fig.text(0.5, 0.02, 'Lower values indicate better performance • ZKPAS demonstrates superior computational efficiency', 
         ha='center', va='bottom', fontsize=10, style='italic', color='#7f8c8d')

plt.tight_layout()
plt.subplots_adjust(bottom=0.15, top=0.9)

# Save the enhanced figure
computational_plot_path = os.path.join(output_dir, "computational_overhead.png")
plt.savefig(computational_plot_path, dpi=300, bbox_inches='tight', 
            facecolor='#f8f9fa', edgecolor='none')
plt.close()

print(f"Enhanced computational overhead graph saved to: {computational_plot_path}")
computational_plot_path

import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set modern style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Define protocols
protocols = ['ZKPAS', '[19]', '[8]', '[15]', '[10]']
output_dir = "."

# Throughput data (authentications/sec) at different entity densities (entities/km²)
densities = list(range(10, 101, 10))
throughput_data = {
    'ZKPAS': [745, 728, 703, 671, 632, 587, 536, 479, 417, 350],
    '[19]': [287, 276, 261, 242, 219, 193, 164, 132, 97, 59],
    '[8]': [421, 408, 391, 369, 343, 313, 279, 241, 199, 153],
    '[15]': [358, 346, 330, 310, 286, 258, 226, 190, 150, 106],
    '[10]': [392, 379, 363, 342, 317, 288, 255, 218, 177, 132]
}

# Create figure with enhanced styling
fig, ax = plt.subplots(figsize=(16, 10))
fig.patch.set_facecolor('#f8f9fa')

# Enhanced color palette
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
markers = ['o', 's', '^', 'D', 'v']
linestyles = ['-', '--', '-.', ':', '-']

# Draw enhanced line plot
for i, protocol in enumerate(protocols):
    ax.plot(densities, throughput_data[protocol], 
            label=protocol, 
            linewidth=3, 
            marker=markers[i], 
            markersize=8,
            color=colors[i],
            linestyle=linestyles[i],
            alpha=0.8,
            markeredgecolor='white',
            markeredgewidth=1.5)

# Enhanced styling
ax.set_xlabel('Entity Density (entities/km²)', fontsize=14, fontweight='bold', color='#2c3e50')
ax.set_ylabel('Authentication Throughput (auth/sec)', fontsize=14, fontweight='bold', color='#2c3e50')
ax.set_title('Authentication Throughput Scaling Under Entity Density\nZero-Knowledge Predictive Authentication System (ZKPAS)', 
             fontsize=16, fontweight='bold', color='#2c3e50', pad=20)

# Enhanced grid
ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.3, color='#7f8c8d')
ax.set_axisbelow(True)

# Enhanced legend
legend = ax.legend(loc='upper right', frameon=True, shadow=True, 
                  fancybox=True, framealpha=0.9, fontsize=11)
legend.get_frame().set_facecolor('#ffffff')
legend.get_frame().set_edgecolor('#bdc3c7')

# Enhanced axis
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.tick_params(axis='both', labelsize=11, color='#34495e')

# Add background gradient effect
ax.set_facecolor('#fdfdfd')

# Add a subtle border
for spine in ax.spines.values():
    spine.set_edgecolor('#bdc3c7')
    spine.set_linewidth(1.5)

# Add performance indicator - highlight ZKPAS superiority
zkpas_min = min(throughput_data['ZKPAS'])
ax.axhline(y=zkpas_min, color='#27ae60', linestyle=':', linewidth=2, alpha=0.7)
ax.text(densities[-1], zkpas_min + 20, 'ZKPAS Minimum Performance', 
        ha='right', va='bottom', fontsize=10, color='#27ae60', fontweight='bold')

# Add subtitle
fig.text(0.5, 0.02, 'Higher values indicate better performance • ZKPAS maintains superior throughput across all density levels', 
         ha='center', va='bottom', fontsize=10, style='italic', color='#7f8c8d')

plt.tight_layout()
plt.subplots_adjust(bottom=0.15, top=0.9)

# Save the enhanced figure
throughput_plot_path = os.path.join(output_dir, "throughput_scaling.png")
plt.savefig(throughput_plot_path, dpi=300, bbox_inches='tight', 
            facecolor='#f8f9fa', edgecolor='none')
plt.close()

print(f"Enhanced throughput scaling graph saved to: {throughput_plot_path}")
throughput_plot_path
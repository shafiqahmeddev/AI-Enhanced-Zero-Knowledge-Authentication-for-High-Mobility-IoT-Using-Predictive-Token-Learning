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

# Define communication overhead data (bytes) for each protocol across phases
communication_data = {
    'ZKPAS': [196, 72, 34, 18, 142, 89],
    '[19]': [368, 164, 0, 56, 288, 0],  # 0 indicates N/A
    '[8]': [276, 132, 0, 44, 224, 0],
    '[15]': [342, 128, 84, 48, 256, 0],
    '[10]': [289, 145, 0, 52, 267, 0]
}

protocols = list(communication_data.keys())
output_dir = "."

phases_comm = ['Initial\nHandshake', 'Authentication\nToken', 'Sliding Window\nLink',
               'Continuous\nValidation', 'Cross-Domain\nTransfer', 'Predictive\nPre-computation']

# Create figure with enhanced styling
fig, ax = plt.subplots(figsize=(16, 10))
fig.patch.set_facecolor('#f8f9fa')

# Enhanced color palette
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
patterns = ['', '///', '...', 'xxx', '|||']

x = np.arange(len(phases_comm))
width = 0.15

# Draw enhanced bar plot
bars = []
for i, protocol in enumerate(protocols):
    offset = (i - 2) * width
    values = [val if val > 0 else np.nan for val in communication_data[protocol]]
    
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
        if not np.isnan(v):
            ax.text(x[j] + offset, v + 5, f'{int(v)}', 
                   ha='center', va='bottom', fontweight='bold', 
                   fontsize=9, color='#2c3e50')
    
    bars.append(bar)

# Enhanced styling
ax.set_xlabel('Authentication Phase', fontsize=14, fontweight='bold', color='#2c3e50')
ax.set_ylabel('Message Size (Bytes)', fontsize=14, fontweight='bold', color='#2c3e50')
ax.set_title('Communication Overhead Across Authentication Phases\nZero-Knowledge Predictive Authentication System (ZKPAS)', 
             fontsize=16, fontweight='bold', color='#2c3e50', pad=20)

# Enhanced x-axis
ax.set_xticks(x)
ax.set_xticklabels(phases_comm, fontsize=11, fontweight='bold', color='#34495e')
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
max_value = max([max([v for v in values if v > 0]) for values in communication_data.values()])
ax.axhline(y=max_value*0.1, color='#27ae60', linestyle=':', linewidth=2, alpha=0.7)
ax.text(len(phases_comm)-1, max_value*0.12, 'Optimal Range', 
        ha='right', va='bottom', fontsize=10, color='#27ae60', fontweight='bold')

# Add subtitle
fig.text(0.5, 0.02, 'Lower values indicate better performance • ZKPAS demonstrates superior efficiency', 
         ha='center', va='bottom', fontsize=10, style='italic', color='#7f8c8d')

plt.tight_layout()
plt.subplots_adjust(bottom=0.15, top=0.9)

# Save the enhanced figure
communication_plot_path = os.path.join(output_dir, "communication_overhead.png")
plt.savefig(communication_plot_path, dpi=300, bbox_inches='tight', 
            facecolor='#f8f9fa', edgecolor='none')
plt.close()

print(f"Enhanced communication overhead graph saved to: {communication_plot_path}")
communication_plot_path

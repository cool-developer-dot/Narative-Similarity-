"""
Create architecture diagram for the LaTeX report
Generates a visual representation of the proposed model

Assignment 3 - CS-272: Artificial Intelligence
National University of Sciences and Technology (NUST)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set publication quality
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'

def create_architecture_diagram():
    """Create architecture diagram for proposed model"""
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define colors
    color_input = '#E8F4F8'
    color_encoder = '#B8E6F0'
    color_pool = '#7DD3E8'
    color_mlp = '#4CB8D8'
    color_loss = '#FFB6C1'
    color_output = '#90EE90'
    
    # Layer 1: Input texts
    y_start = 9
    
    # Input boxes
    input1 = FancyBboxPatch((0.5, y_start), 2, 0.5, 
                            boxstyle="round,pad=0.05", 
                            facecolor=color_input, edgecolor='black', linewidth=1.5)
    ax.add_patch(input1)
    ax.text(1.5, y_start+0.25, 'Anchor Text', ha='center', va='center', fontsize=9, weight='bold')
    
    input2 = FancyBboxPatch((3.5, y_start), 2, 0.5, 
                            boxstyle="round,pad=0.05", 
                            facecolor=color_input, edgecolor='black', linewidth=1.5)
    ax.add_patch(input2)
    ax.text(4.5, y_start+0.25, 'Text A', ha='center', va='center', fontsize=9, weight='bold')
    
    input3 = FancyBboxPatch((6.5, y_start), 2, 0.5, 
                            boxstyle="round,pad=0.05", 
                            facecolor=color_input, edgecolor='black', linewidth=1.5)
    ax.add_patch(input3)
    ax.text(7.5, y_start+0.25, 'Text B', ha='center', va='center', fontsize=9, weight='bold')
    
    # Layer 2: Concatenation
    y_concat = 7.8
    
    concat1 = FancyBboxPatch((1, y_concat), 3, 0.6, 
                            boxstyle="round,pad=0.05", 
                            facecolor=color_encoder, edgecolor='black', linewidth=1.5)
    ax.add_patch(concat1)
    ax.text(2.5, y_concat+0.3, '[CLS] Anchor [SEP] Text A [SEP]', 
            ha='center', va='center', fontsize=8)
    
    concat2 = FancyBboxPatch((5.5, y_concat), 3, 0.6, 
                            boxstyle="round,pad=0.05", 
                            facecolor=color_encoder, edgecolor='black', linewidth=1.5)
    ax.add_patch(concat2)
    ax.text(7, y_concat+0.3, '[CLS] Anchor [SEP] Text B [SEP]', 
            ha='center', va='center', fontsize=8)
    
    # Arrows from input to concat
    arrow1 = FancyArrowPatch((1.5, y_start), (2.5, y_concat+0.6),
                            arrowstyle='->', lw=2, color='gray')
    ax.add_patch(arrow1)
    arrow2 = FancyArrowPatch((4.5, y_start), (2.5, y_concat+0.6),
                            arrowstyle='->', lw=2, color='gray')
    ax.add_patch(arrow2)
    arrow3 = FancyArrowPatch((7.5, y_start), (7, y_concat+0.6),
                            arrowstyle='->', lw=2, color='gray')
    ax.add_patch(arrow3)
    
    # Layer 3: Shared Encoder
    y_encoder = 6.2
    
    encoder1 = FancyBboxPatch((1, y_encoder), 3, 1, 
                             boxstyle="round,pad=0.05", 
                             facecolor=color_pool, edgecolor='black', linewidth=2)
    ax.add_patch(encoder1)
    ax.text(2.5, y_encoder+0.7, 'MPNet Encoder', ha='center', va='center', 
            fontsize=9, weight='bold')
    ax.text(2.5, y_encoder+0.35, '(12 Transformer Layers)', ha='center', va='center', 
            fontsize=7, style='italic')
    
    encoder2 = FancyBboxPatch((5.5, y_encoder), 3, 1, 
                             boxstyle="round,pad=0.05", 
                             facecolor=color_pool, edgecolor='black', linewidth=2)
    ax.add_patch(encoder2)
    ax.text(7, y_encoder+0.7, 'MPNet Encoder', ha='center', va='center', 
            fontsize=9, weight='bold')
    ax.text(7, y_encoder+0.35, '(Shared Weights)', ha='center', va='center', 
            fontsize=7, style='italic')
    
    # Arrows
    arrow4 = FancyArrowPatch((2.5, y_concat), (2.5, y_encoder+1),
                            arrowstyle='->', lw=2.5, color='darkblue')
    ax.add_patch(arrow4)
    arrow5 = FancyArrowPatch((7, y_concat), (7, y_encoder+1),
                            arrowstyle='->', lw=2.5, color='darkblue')
    ax.add_patch(arrow5)
    
    # Layer 4: Pooling
    y_pool = 4.8
    
    pool1 = FancyBboxPatch((1, y_pool), 3, 0.6, 
                          boxstyle="round,pad=0.05", 
                          facecolor=color_mlp, edgecolor='black', linewidth=1.5)
    ax.add_patch(pool1)
    ax.text(2.5, y_pool+0.3, '[CLS] + Mean Pooling', ha='center', va='center', fontsize=8)
    
    pool2 = FancyBboxPatch((5.5, y_pool), 3, 0.6, 
                          boxstyle="round,pad=0.05", 
                          facecolor=color_mlp, edgecolor='black', linewidth=1.5)
    ax.add_patch(pool2)
    ax.text(7, y_pool+0.3, '[CLS] + Mean Pooling', ha='center', va='center', fontsize=8)
    
    # Arrows
    arrow6 = FancyArrowPatch((2.5, y_encoder), (2.5, y_pool+0.6),
                            arrowstyle='->', lw=2, color='darkgreen')
    ax.add_patch(arrow6)
    arrow7 = FancyArrowPatch((7, y_encoder), (7, y_pool+0.6),
                            arrowstyle='->', lw=2, color='darkgreen')
    ax.add_patch(arrow7)
    
    # Layer 5: MLP Scoring Head
    y_mlp = 3.2
    
    mlp1 = FancyBboxPatch((1.2, y_mlp), 2.6, 0.8, 
                         boxstyle="round,pad=0.05", 
                         facecolor=color_mlp, edgecolor='black', linewidth=1.5)
    ax.add_patch(mlp1)
    ax.text(2.5, y_mlp+0.5, 'MLP Scoring Head', ha='center', va='center', 
            fontsize=8, weight='bold')
    ax.text(2.5, y_mlp+0.15, 'FC(768→512→256→1)', ha='center', va='center', fontsize=7)
    
    mlp2 = FancyBboxPatch((5.7, y_mlp), 2.6, 0.8, 
                         boxstyle="round,pad=0.05", 
                         facecolor=color_mlp, edgecolor='black', linewidth=1.5)
    ax.add_patch(mlp2)
    ax.text(7, y_mlp+0.5, 'MLP Scoring Head', ha='center', va='center', 
            fontsize=8, weight='bold')
    ax.text(7, y_mlp+0.15, 'FC(768→512→256→1)', ha='center', va='center', fontsize=7)
    
    # Arrows
    arrow8 = FancyArrowPatch((2.5, y_pool), (2.5, y_mlp+0.8),
                            arrowstyle='->', lw=2, color='purple')
    ax.add_patch(arrow8)
    arrow9 = FancyArrowPatch((7, y_pool), (7, y_mlp+0.8),
                            arrowstyle='->', lw=2, color='purple')
    ax.add_patch(arrow9)
    
    # Layer 6: Scores
    y_scores = 2
    
    score1 = FancyBboxPatch((1.5, y_scores), 2, 0.5, 
                           boxstyle="round,pad=0.05", 
                           facecolor=color_output, edgecolor='black', linewidth=1.5)
    ax.add_patch(score1)
    ax.text(2.5, y_scores+0.25, 'Score A', ha='center', va='center', 
            fontsize=9, weight='bold')
    
    score2 = FancyBboxPatch((6, y_scores), 2, 0.5, 
                           boxstyle="round,pad=0.05", 
                           facecolor=color_output, edgecolor='black', linewidth=1.5)
    ax.add_patch(score2)
    ax.text(7, y_scores+0.25, 'Score B', ha='center', va='center', 
            fontsize=9, weight='bold')
    
    # Arrows
    arrow10 = FancyArrowPatch((2.5, y_mlp), (2.5, y_scores+0.5),
                             arrowstyle='->', lw=2.5, color='darkred')
    ax.add_patch(arrow10)
    arrow11 = FancyArrowPatch((7, y_mlp), (7, y_scores+0.5),
                             arrowstyle='->', lw=2.5, color='darkred')
    ax.add_patch(arrow11)
    
    # Layer 7: Loss Functions
    y_loss = 0.5
    
    loss_box = FancyBboxPatch((2, y_loss), 5, 0.8, 
                             boxstyle="round,pad=0.05", 
                             facecolor=color_loss, edgecolor='black', linewidth=2)
    ax.add_patch(loss_box)
    ax.text(4.5, y_loss+0.5, 'Multi-Task Loss', ha='center', va='center', 
            fontsize=9, weight='bold')
    ax.text(4.5, y_loss+0.15, 'L = (1-α)·L_cls + α·L_triplet', ha='center', va='center', 
            fontsize=7, style='italic')
    
    # Arrows to loss
    arrow12 = FancyArrowPatch((2.5, y_scores), (3.5, y_loss+0.8),
                             arrowstyle='->', lw=2, color='red')
    ax.add_patch(arrow12)
    arrow13 = FancyArrowPatch((7, y_scores), (5.5, y_loss+0.8),
                             arrowstyle='->', lw=2, color='red')
    ax.add_patch(arrow13)
    
    # Add side annotations
    ax.text(9.2, y_start+0.25, 'Input', fontsize=9, weight='bold', rotation=90, va='center')
    ax.text(9.2, y_encoder+0.5, 'Encoding', fontsize=9, weight='bold', rotation=90, va='center')
    ax.text(9.2, y_mlp+0.4, 'Scoring', fontsize=9, weight='bold', rotation=90, va='center')
    ax.text(9.2, y_loss+0.4, 'Learning', fontsize=9, weight='bold', rotation=90, va='center')
    
    # Title
    ax.text(5, 9.8, 'Hybrid Cross-Encoder Architecture with Contrastive Learning', 
            ha='center', va='center', fontsize=12, weight='bold')
    
    # Add legend for components
    legend_elements = [
        mpatches.Patch(facecolor=color_input, edgecolor='black', label='Input Layer'),
        mpatches.Patch(facecolor=color_encoder, edgecolor='black', label='Tokenization'),
        mpatches.Patch(facecolor=color_pool, edgecolor='black', label='Transformer'),
        mpatches.Patch(facecolor=color_mlp, edgecolor='black', label='MLP/Pooling'),
        mpatches.Patch(facecolor=color_output, edgecolor='black', label='Output Scores'),
        mpatches.Patch(facecolor=color_loss, edgecolor='black', label='Loss Function')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=7, 
              framealpha=0.9, bbox_to_anchor=(0, 0.95))
    
    plt.tight_layout()
    plt.savefig('architecture_diagram.pdf', format='pdf', bbox_inches='tight')
    print("✓ Architecture diagram saved to: architecture_diagram.pdf")
    plt.close()


if __name__ == "__main__":
    create_architecture_diagram()
    print("\nTo use in LaTeX report:")
    print("  1. Place 'architecture_diagram.pdf' in the same directory as .tex file")
    print("  2. The diagram is already referenced in Figure 1 of the report")
    print("  3. Compile the LaTeX document")


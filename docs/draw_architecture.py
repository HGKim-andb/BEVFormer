"""
Generate visual architecture diagram for Risk-Guided BEVFormer
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np


def draw_architecture():
    """Draw complete architecture diagram"""

    fig, ax = plt.subplots(figsize=(20, 28))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 28)
    ax.axis('off')

    # Color scheme
    color_input = '#E8F4F8'
    color_backbone = '#B8E6F0'
    color_transformer = '#88D8E8'
    color_bev = '#FFF9E6'
    color_detection = '#FFE6CC'
    color_risk = '#FFE6E6'
    color_loss = '#E6F5E6'

    # Title
    ax.text(10, 27.5, 'Risk-Guided BEVFormer Architecture',
            fontsize=24, fontweight='bold', ha='center')

    y_pos = 26.5

    # ============================================================
    # 1. INPUT: Multi-Camera Images
    # ============================================================
    draw_box(ax, 2, y_pos-1, 16, 0.8, 'Multi-Camera Input (6 cameras)\n[B, 6, 3, H, W]',
             color_input, fontsize=12, bold=True)

    # Draw 6 small camera icons
    cam_positions = [4, 6, 8, 10, 12, 14]
    for i, x in enumerate(cam_positions):
        draw_small_box(ax, x-0.3, y_pos-1.8, 0.6, 0.5,
                      f'C{i+1}', color_input, fontsize=8)

    draw_arrow(ax, 10, y_pos-2.5, 10, y_pos-3)
    y_pos -= 3.5

    # ============================================================
    # 2. IMAGE BACKBONE
    # ============================================================
    draw_box(ax, 2, y_pos-1.5, 16, 1.5,
             'Image Backbone (ResNet-50/101)',
             color_backbone, fontsize=12, bold=True)

    # Conv blocks
    conv_blocks = ['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4']
    for i, name in enumerate(conv_blocks):
        x = 3.5 + i * 3.5
        draw_small_box(ax, x, y_pos-1.2, 2, 0.8, name, color_backbone, fontsize=9)
        if i < 3:
            ax.annotate('', xy=(x+2.5, y_pos-0.8), xytext=(x+2.1, y_pos-0.8),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    ax.text(10, y_pos-1.7, 'Output: [B×6, 256, H/32, W/32]',
            fontsize=9, ha='center', style='italic')

    draw_arrow(ax, 10, y_pos-2.2, 10, y_pos-2.7)
    y_pos -= 3.5

    # ============================================================
    # 3. FPN NECK
    # ============================================================
    draw_box(ax, 2, y_pos-1, 16, 1,
             'Feature Pyramid Network (FPN)',
             color_backbone, fontsize=11, bold=True)

    fpn_scales = ['H/8', 'H/16', 'H/32', 'H/64']
    for i, scale in enumerate(fpn_scales):
        ax.text(4 + i*3, y_pos-0.7, f'[B×6, 256, {scale}]',
               fontsize=8, ha='center')

    draw_arrow(ax, 10, y_pos-1.5, 10, y_pos-2)
    y_pos -= 2.5

    # ============================================================
    # 4. BEV TRANSFORMER ENCODER
    # ============================================================
    draw_box(ax, 1.5, y_pos-5, 17, 5,
             'BEV Transformer Encoder',
             color_transformer, fontsize=12, bold=True)

    # BEV Query Init
    draw_box(ax, 2.5, y_pos-1.2, 15, 0.8,
             'BEV Query Initialization [H_bev × W_bev, B, C] = [2500, B, 256]',
             'white', fontsize=10)

    # Encoder layers
    encoder_y = y_pos - 2.2
    for layer_idx in range(2):  # Show 2 layers as example
        layer_y = encoder_y - layer_idx * 2.2

        # Layer box
        draw_box(ax, 3, layer_y-1.8, 14, 1.8,
                f'Encoder Layer {layer_idx + 1}',
                '#D0E8F0', fontsize=9, bold=True)

        # Three components
        components = [
            ('Temporal\nSelf-Attn', layer_y-0.6),
            ('Spatial\nCross-Attn', layer_y-1.1),
            ('FFN', layer_y-1.6)
        ]

        for i, (comp_name, comp_y) in enumerate(components):
            draw_small_box(ax, 4 + i*3.5, comp_y, 3, 0.4, comp_name, 'white', fontsize=7)
            if i < 2:
                ax.annotate('', xy=(7.6 + i*3.5, comp_y+0.05),
                          xytext=(7.1 + i*3.5, comp_y+0.05),
                          arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

    ax.text(17.5, y_pos-3, '×6 layers', fontsize=9, rotation=90, va='center')

    # Output
    ax.text(10, y_pos-5.3, 'BEV Features: [B, H_bev×W_bev, C] = [B, 2500, 256]',
            fontsize=10, ha='center', style='italic', fontweight='bold')

    draw_arrow(ax, 10, y_pos-5.7, 10, y_pos-6.2)
    y_pos -= 7

    # ============================================================
    # 5. BEV FEATURES (Shared)
    # ============================================================
    draw_box(ax, 3, y_pos-1, 14, 1,
             'Shared BEV Representation\n[B, 256, 50, 50] or [B, 2500, 256]',
             color_bev, fontsize=11, bold=True)

    # Split arrows
    draw_arrow(ax, 7, y_pos-1.5, 7, y_pos-2, color='blue', lw=3)
    draw_arrow(ax, 13, y_pos-1.5, 13, y_pos-2, color='red', lw=3)

    y_pos -= 2.5

    # ============================================================
    # 6. DETECTION PATH (Left) & RISK PATH (Right)
    # ============================================================

    # DETECTION PATH
    det_x = 4
    draw_box(ax, det_x-1.5, y_pos-5, 6, 5,
             'DETECTION PATH', color_detection, fontsize=11, bold=True)

    # Transformer Decoder
    draw_box(ax, det_x-1, y_pos-1.5, 5, 1.2,
             'Transformer Decoder\n(6 layers)', 'white', fontsize=9, bold=True)

    ax.text(det_x+1.5, y_pos-2, 'Object Queries: [B, 900, 256]',
           fontsize=8, ha='center')
    ax.text(det_x+1.5, y_pos-2.3, '↓ Deformable Attention', fontsize=7, ha='center')
    ax.text(det_x+1.5, y_pos-2.6, '↓ Self-Attention', fontsize=7, ha='center')
    ax.text(det_x+1.5, y_pos-2.9, '↓ FFN', fontsize=7, ha='center')

    # Detection heads
    draw_small_box(ax, det_x-0.5, y_pos-3.7, 2, 0.5, 'Cls Head', '#FFD9B3', fontsize=8)
    draw_small_box(ax, det_x+2, y_pos-3.7, 2, 0.5, 'Reg Head', '#FFD9B3', fontsize=8)

    # Outputs
    ax.text(det_x+1.5, y_pos-4.4, 'Classes: [B, 900, 10]', fontsize=7, ha='center')
    ax.text(det_x+1.5, y_pos-4.7, 'Boxes: [B, 900, 10]', fontsize=7, ha='center')

    # RISK PATH
    risk_x = 13
    draw_box(ax, risk_x-1.5, y_pos-5, 6, 5,
             'RISK PATH (NEW!)', color_risk, fontsize=11, bold=True)

    # Risk head layers
    risk_layers = [
        ('Conv 256→128\n+ BN + ReLU', y_pos-1.2),
        ('Conv 128→128\n+ BN + ReLU', y_pos-2),
        ('Conv 128→64\n+ BN + ReLU', y_pos-2.8),
        ('Conv 64→1', y_pos-3.6),
        ('Upsample\n50×50 → 200×200', y_pos-4.4),
    ]

    for i, (layer_name, layer_y) in enumerate(risk_layers):
        draw_small_box(ax, risk_x-1, layer_y, 5, 0.6, layer_name, 'white', fontsize=7)
        if i < 4:
            draw_arrow(ax, risk_x+1.5, layer_y-0.35, risk_x+1.5, layer_y-0.55, lw=1.5)

    draw_arrow(ax, det_x+1.5, y_pos-5.3, det_x+1.5, y_pos-5.8, color='blue', lw=2)
    draw_arrow(ax, risk_x+1.5, y_pos-5.3, risk_x+1.5, y_pos-5.8, color='red', lw=2)

    y_pos -= 6.5

    # ============================================================
    # 7. OUTPUTS
    # ============================================================

    # Detection output
    draw_box(ax, det_x-1.5, y_pos-1.5, 6, 1.5,
             'Detection Output\n\n• 3D Bboxes\n• Classes\n• Scores\n• Velocity',
             color_detection, fontsize=9)

    # Risk output
    draw_box(ax, risk_x-1.5, y_pos-1.5, 6, 1.5,
             'Risk Map Output\n\n[B, 1, 200, 200]\nValues ∈ [0, 1]\n0.5m resolution',
             color_risk, fontsize=9)

    draw_arrow(ax, det_x+1.5, y_pos-2, det_x+1.5, y_pos-2.5, color='blue', lw=2)
    draw_arrow(ax, risk_x+1.5, y_pos-2, risk_x+1.5, y_pos-2.5, color='red', lw=2)

    y_pos -= 3

    # ============================================================
    # 8. LOSS COMPUTATION
    # ============================================================

    draw_box(ax, 2, y_pos-2.5, 16, 2.5,
             'Multi-Task Loss Computation', color_loss, fontsize=12, bold=True)

    # Detection loss
    ax.text(6, y_pos-1, 'L_detection = L_cls + L_bbox',
           fontsize=10, ha='center', family='monospace',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Risk loss
    ax.text(14, y_pos-1, 'L_risk = L_mse + 0.5 × L_mae',
           fontsize=10, ha='center', family='monospace',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Total loss
    ax.text(10, y_pos-2, 'L_total = L_detection + λ × L_risk',
           fontsize=11, ha='center', family='monospace', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='#FFFFCC', alpha=0.9))

    # ============================================================
    # Legend
    # ============================================================
    legend_y = 1
    ax.text(10, legend_y + 0.5, 'Component Legend', fontsize=11, ha='center', fontweight='bold')

    legend_items = [
        (color_input, 'Input Data'),
        (color_backbone, 'Backbone/Neck'),
        (color_transformer, 'Transformer'),
        (color_bev, 'BEV Features'),
        (color_detection, 'Detection'),
        (color_risk, 'Risk (NEW)'),
        (color_loss, 'Loss'),
    ]

    for i, (color, label) in enumerate(legend_items):
        x = 3 + (i % 4) * 4
        y = legend_y - 0.3 - (i // 4) * 0.5
        rect = mpatches.Rectangle((x, y-0.15), 0.4, 0.3,
                                  facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(x + 0.6, y, label, fontsize=9, va='center')

    plt.tight_layout()
    return fig


def draw_box(ax, x, y, width, height, text, color, fontsize=10, bold=False):
    """Draw a colored box with text"""
    weight = 'bold' if bold else 'normal'
    rect = FancyBboxPatch((x, y), width, height,
                          boxstyle="round,pad=0.1",
                          facecolor=color,
                          edgecolor='black',
                          linewidth=2)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, text,
           fontsize=fontsize, ha='center', va='center', fontweight=weight)


def draw_small_box(ax, x, y, width, height, text, color, fontsize=8):
    """Draw a smaller box"""
    rect = mpatches.Rectangle((x, y), width, height,
                             facecolor=color, edgecolor='black', linewidth=1)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, text,
           fontsize=fontsize, ha='center', va='center')


def draw_arrow(ax, x1, y1, x2, y2, color='black', lw=2):
    """Draw an arrow"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
               arrowprops=dict(arrowstyle='->', lw=lw, color=color))


def draw_bev_grid():
    """Draw BEV grid coordinate system"""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Grid
    ax.set_xlim(-55, 55)
    ax.set_ylim(-55, 55)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(0, color='black', linewidth=2)
    ax.axvline(0, color='black', linewidth=2)

    # Labels
    ax.set_xlabel('X (Forward) [meters]', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y (Left) [meters]', fontsize=14, fontweight='bold')
    ax.set_title('BEV Coordinate System\n200×200 grid, 0.5m resolution',
                fontsize=16, fontweight='bold')

    # Ego vehicle
    ego = Circle((0, 0), 2, color='cyan', ec='black', linewidth=2, zorder=10)
    ax.add_patch(ego)
    ax.text(0, 0, 'EGO', ha='center', va='center', fontweight='bold', fontsize=10)

    # Forward arrow
    ax.arrow(0, 0, 0, -15, head_width=3, head_length=3,
            fc='cyan', ec='black', linewidth=2, zorder=9)
    ax.text(0, -20, 'FORWARD', ha='center', fontsize=12, fontweight='bold')

    # Quadrants
    quadrants = [
        (25, 25, 'Q1\n(+X, +Y)', '#FFE6E6'),
        (-25, 25, 'Q2\n(-X, +Y)', '#E6FFE6'),
        (-25, -25, 'Q3\n(-X, -Y)', '#E6E6FF'),
        (25, -25, 'Q4\n(+X, -Y)', '#FFFFE6'),
    ]

    for qx, qy, label, color in quadrants:
        rect = mpatches.Rectangle((qx-25, qy-25), 50, 50,
                                 facecolor=color, alpha=0.3, edgecolor='gray')
        ax.add_patch(rect)
        ax.text(qx, qy, label, ha='center', va='center',
               fontsize=11, fontweight='bold')

    # Range annotations
    ax.text(50, -52, '+50m', ha='center', fontsize=10)
    ax.text(-50, -52, '-50m', ha='center', fontsize=10)
    ax.text(-52, 50, '+50m', va='center', fontsize=10, rotation=90)
    ax.text(-52, -50, '-50m', va='center', fontsize=10, rotation=90)

    # Pixel coordinate note
    note_text = ('Pixel Coordinates:\n'
                '• (0, 0) = Top-left = (-50m, -50m)\n'
                '• (100, 100) = Center = Ego (0m, 0m)\n'
                '• (199, 199) = Bottom-right = (+50m, +50m)')
    ax.text(-50, 45, note_text, fontsize=9,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout()
    return fig


def draw_risk_guided_attention():
    """Draw risk-guided attention mechanism"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(8, 9.5, 'Risk-Guided Attention Mechanism',
           fontsize=18, fontweight='bold', ha='center')

    y = 8.5

    # Input BEV features
    draw_box(ax, 1, y-0.8, 14, 0.8,
            'BEV Features [B, 256, 50, 50]', '#FFF9E6', fontsize=11, bold=True)

    # Split into three paths
    y = 7
    draw_arrow(ax, 4, y+0.5, 4, y, lw=2)
    draw_arrow(ax, 8, y+0.5, 8, y, lw=2)
    draw_arrow(ax, 12, y+0.5, 12, y, lw=2)

    # Path 1: Standard conv
    draw_box(ax, 2, y-1, 4, 1, 'Standard\nConv Layers', '#B8E6F0', fontsize=10)

    # Path 2: Risk prediction
    draw_box(ax, 6.5, y-1, 3, 1, 'Risk\nPrediction', '#FFE6E6', fontsize=10)
    draw_arrow(ax, 8, y-1.3, 8, y-1.8, lw=2)
    draw_box(ax, 6.5, y-2.5, 3, 0.7,
            'Risk Map\n[B, 1, 200, 200]', 'white', fontsize=9)

    # Path 3: Attention generator
    draw_box(ax, 10.5, y-1, 3.5, 1, 'Spatial Attention\nGenerator', '#E6F5E6', fontsize=10)

    # Downsample
    y = 4.5
    draw_arrow(ax, 8, y+0.7, 8, y+0.2, lw=2)
    draw_box(ax, 6.5, y-0.5, 3, 0.5,
            'Downsample\n200×200 → 50×50', 'white', fontsize=8)

    # Merge paths
    y = 3.5
    draw_arrow(ax, 8, y+0.5, 8, y, lw=2)
    draw_box(ax, 6, y-1, 4, 1,
            'Attention Weight\nGeneration', '#D0E8F0', fontsize=10, bold=True)

    ax.text(8, y-0.5, 'Conv(1→32→1) + Sigmoid',
           ha='center', fontsize=8, style='italic')

    # Attention weights
    y = 2
    draw_arrow(ax, 8, y+0.5, 8, y, lw=2)
    draw_box(ax, 6.5, y-0.8, 3, 0.8,
            'Attention Weights\n[B, 1, 50, 50]', '#FFFFCC', fontsize=10, bold=True)

    # Element-wise multiplication
    y = 0.7
    draw_arrow(ax, 4, 1.2, 6.5, y, lw=2, color='blue')
    draw_arrow(ax, 8, y+0.5, 8, y, lw=2, color='red')

    ax.text(8, y, '⊗', ha='center', va='center', fontsize=30, fontweight='bold')

    # Output
    y = 0
    draw_box(ax, 5, y-0.8, 6, 0.8,
            'Risk-Attended Features\n[B, 256, 50, 50]', '#E6FFE6',
            fontsize=11, bold=True)

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    # Generate all diagrams
    print("Generating architecture diagram...")
    fig1 = draw_architecture()
    fig1.savefig('docs/architecture_full.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: docs/architecture_full.png")

    print("\nGenerating BEV grid diagram...")
    fig2 = draw_bev_grid()
    fig2.savefig('docs/bev_grid.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: docs/bev_grid.png")

    print("\nGenerating risk-guided attention diagram...")
    fig3 = draw_risk_guided_attention()
    fig3.savefig('docs/risk_attention.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: docs/risk_attention.png")

    print("\n🎉 All diagrams generated successfully!")
    print("\nTo view the diagrams, run:")
    print("  python docs/draw_architecture.py")
    print("\nOr open the PNG files directly:")
    print("  - docs/architecture_full.png")
    print("  - docs/bev_grid.png")
    print("  - docs/risk_attention.png")

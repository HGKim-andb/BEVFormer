#!/usr/bin/env python3
"""
Manual Emergence Labeling Tool

Interactive GUI for manually labeling emergence events in nuScenes dataset.
Use keyboard and mouse to navigate scenes and label emergences on BEV space.
"""

import numpy as np
import pickle
import argparse
from pathlib import Path
import cv2
import matplotlib
matplotlib.use('TkAgg')  # Interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
import sys

try:
    from nuscenes.nuscenes import NuScenes
    from pyquaternion import Quaternion
except ImportError:
    print("Error: nuscenes-devkit not installed. Install with: pip install nuscenes-devkit pyquaternion")
    sys.exit(1)


class EmergenceLabelingTool:
    """Interactive tool for manual emergence labeling"""

    def __init__(self, dataroot, version='v1.0-mini', output_path='data/manual_labels.pkl'):
        self.dataroot = Path(dataroot)
        self.version = version
        self.output_path = Path(output_path)

        # Load nuScenes
        print(f"Loading nuScenes {version}...")
        self.nusc = NuScenes(version=version, dataroot=str(self.dataroot), verbose=False)

        # Get all scenes
        self.scenes = self.nusc.scene
        self.current_scene_idx = 0
        self.current_sample_idx = 0

        # Labels storage: {scene_token: {sample_token: [(x, y), ...]}}
        self.labels = {}

        # BEV configuration
        self.bev_range = [-50, 50, -50, 50]  # [x_min, x_max, y_min, y_max]

        # UI state
        self.selected_point = None  # For removal

        # Load scene samples (must be after self.labels initialization)
        self.load_scene_samples()

        # Create GUI
        self.create_gui()

        # Load initial view
        self.update_view()

    def load_scene_samples(self):
        """Load all samples for current scene"""
        scene = self.scenes[self.current_scene_idx]
        scene_token = scene['token']

        # Get all samples in chronological order
        sample_tokens = []
        sample_token = scene['first_sample_token']

        while sample_token:
            sample_tokens.append(sample_token)
            sample = self.nusc.get('sample', sample_token)
            sample_token = sample['next']

        self.current_scene_token = scene_token
        self.current_scene_name = scene['name']
        self.sample_tokens = sample_tokens
        self.current_sample_idx = 0

        # Initialize labels for this scene if not exists
        if scene_token not in self.labels:
            self.labels[scene_token] = {}

    def create_gui(self):
        """Create the main GUI window"""
        self.root = tk.Tk()
        self.root.title("Emergence Manual Labeling Tool")
        self.root.geometry("1800x1200")

        # Top control panel
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # Scene selector
        ttk.Label(control_frame, text="Scene:").pack(side=tk.LEFT, padx=5)
        self.scene_var = tk.StringVar(value=f"{self.current_scene_idx + 1}/{len(self.scenes)}")
        self.scene_label = ttk.Label(control_frame, textvariable=self.scene_var,
                                     font=('Arial', 10, 'bold'))
        self.scene_label.pack(side=tk.LEFT, padx=5)

        ttk.Button(control_frame, text="◀ Prev Scene",
                  command=self.prev_scene).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Next Scene ▶",
                  command=self.next_scene).pack(side=tk.LEFT, padx=5)

        # Sample/time selector
        ttk.Label(control_frame, text="  |  Sample:").pack(side=tk.LEFT, padx=5)
        self.sample_var = tk.StringVar(value=f"{self.current_sample_idx + 1}/{len(self.sample_tokens)}")
        self.sample_label = ttk.Label(control_frame, textvariable=self.sample_var,
                                      font=('Arial', 10, 'bold'))
        self.sample_label.pack(side=tk.LEFT, padx=5)

        ttk.Button(control_frame, text="◀ Prev (←)",
                  command=self.prev_sample).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Next (→) ▶",
                  command=self.next_sample).pack(side=tk.LEFT, padx=5)

        # Label count
        ttk.Label(control_frame, text="  |  Labels:").pack(side=tk.LEFT, padx=5)
        self.label_count_var = tk.StringVar(value="0")
        ttk.Label(control_frame, textvariable=self.label_count_var,
                 font=('Arial', 10, 'bold'), foreground='red').pack(side=tk.LEFT, padx=5)

        # Save/Load buttons
        ttk.Button(control_frame, text="💾 Save",
                  command=self.save_labels,
                  style='Save.TButton').pack(side=tk.RIGHT, padx=5)
        ttk.Button(control_frame, text="📂 Load",
                  command=self.load_labels).pack(side=tk.RIGHT, padx=5)

        # Instructions
        info_frame = ttk.Frame(self.root)
        info_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        instructions = (
            "📌 Instructions: LEFT CLICK on BEV to add label | "
            "RIGHT CLICK on label to remove | "
            "← → to navigate samples | "
            "ESC to clear selection"
        )
        ttk.Label(info_frame, text=instructions,
                 foreground='blue', font=('Arial', 9)).pack()

        # Main canvas for matplotlib figure
        self.fig = plt.figure(figsize=(18, 12))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Bind events
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.root.bind('<Left>', lambda e: self.prev_sample())
        self.root.bind('<Right>', lambda e: self.next_sample())
        self.root.bind('<Escape>', lambda e: self.clear_selection())
        self.root.bind('q', lambda e: self.quit_app())

        # Configure button style
        style = ttk.Style()
        style.configure('Save.TButton', font=('Arial', 10, 'bold'))

    def update_view(self):
        """Update the entire view with current sample"""
        self.fig.clear()

        # Get current sample
        sample_token = self.sample_tokens[self.current_sample_idx]
        sample = self.nusc.get('sample', sample_token)

        # Update labels
        self.scene_var.set(f"{self.current_scene_idx + 1}/{len(self.scenes)} - {self.current_scene_name}")
        self.sample_var.set(f"{self.current_sample_idx + 1}/{len(self.sample_tokens)}")

        # Count labels for current sample
        current_labels = self.labels[self.current_scene_token].get(sample_token, [])
        self.label_count_var.set(str(len(current_labels)))

        # Layout: 6 rows x 3 cols
        # Row 0: [FRONT_LEFT] [FRONT] [FRONT_RIGHT]
        # Row 1-3: [======== BEV (large, 3 rows) ========]
        # Row 4: [BACK_LEFT] [BACK] [BACK_RIGHT]

        camera_positions = {
            'CAM_FRONT_LEFT': (0, 0, 1, 1),      # (row, col, rowspan, colspan)
            'CAM_FRONT': (0, 1, 1, 1),
            'CAM_FRONT_RIGHT': (0, 2, 1, 1),
            'CAM_BACK_LEFT': (4, 0, 1, 1),
            'CAM_BACK': (4, 1, 1, 1),
            'CAM_BACK_RIGHT': (4, 2, 1, 1),
        }

        # Draw cameras
        for cam_name, (row, col, rowspan, colspan) in camera_positions.items():
            ax = plt.subplot2grid((6, 3), (row, col), rowspan=rowspan, colspan=colspan)

            cam_token = sample['data'][cam_name]
            cam_data = self.nusc.get('sample_data', cam_token)
            img_path = self.dataroot / cam_data['filename']

            if img_path.exists():
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, f'{cam_name}\nNot found',
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=10)

            # Camera name as title
            cam_display_name = cam_name.replace('CAM_', '').replace('_', ' ')
            ax.set_title(cam_display_name, fontsize=9, fontweight='bold')
            ax.axis('off')

        # Draw BEV (center position - rows 1-3, much larger)
        self.ax_bev = plt.subplot2grid((6, 3), (1, 0), rowspan=3, colspan=3)
        self.draw_bev(sample, current_labels)

        # Main title
        self.fig.suptitle(
            f'Scene: {self.current_scene_name}  |  Sample: {self.current_sample_idx + 1}/{len(self.sample_tokens)}  |  Token: {sample_token[:12]}...',
            fontsize=12, fontweight='bold'
        )

        plt.tight_layout()
        self.canvas.draw()

    def draw_bev(self, sample, labels):
        """Draw BEV space with labels"""
        ax = self.ax_bev
        ax.clear()
        # Coordinate system: forward direction points up (12 o'clock)
        # X axis (plot): left(-50) to right(+50)
        # Y axis (plot): backward(-50) to forward(+50) pointing up
        ax.set_xlim(self.bev_range[2], self.bev_range[3])  # -50 to 50 (left to right)
        ax.set_ylim(self.bev_range[0], self.bev_range[1])  # -50 to 50 (back to forward)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_xlabel('← Left | Right → (meters)', fontsize=10)
        ax.set_ylabel('↑ Forward Direction (meters)', fontsize=10)
        ax.set_title('Bird\'s Eye View - Click to Label (Forward is UP ↑)', fontsize=11, fontweight='bold')

        # Draw ego vehicle (blue rectangle at origin, rotated for 12 o'clock orientation)
        from matplotlib.patches import Rectangle
        # Swap width/height for rotated view: forward(x) -> up(y), left(y) -> right(-x)
        ego_rect = Rectangle((-1, -2), 2, 4, linewidth=2,
                            edgecolor='blue', facecolor='lightblue', alpha=0.6, zorder=10)
        ax.add_patch(ego_rect)
        ax.text(0, 0, 'EGO', ha='center', va='center', fontsize=8,
               fontweight='bold', color='darkblue', zorder=11)

        # Draw current detections (green dots - small)
        sample_token = sample['token']
        ego_trans, ego_rot = self.get_ego_pose(sample_token)

        for ann_token in sample['anns']:
            ann = self.nusc.get('sample_annotation', ann_token)
            global_pos = ann['translation']
            x_ego, y_ego, z = self.global_to_ego(global_pos, ego_trans, ego_rot)

            # Only show objects within BEV range
            if self.bev_range[0] <= x_ego <= self.bev_range[1] and \
               self.bev_range[2] <= y_ego <= self.bev_range[3]:
                # Transform: x_ego (forward) -> y_plot (up), y_ego (left +) -> x_plot (left -)
                # Negate y_ego because: y_ego>0 is left, but x_plot<0 should be left on screen
                x_plot, y_plot = -y_ego, x_ego
                circle = Circle((x_plot, y_plot), 0.8, color='green', alpha=0.3, zorder=5)
                ax.add_patch(circle)
                ax.plot(x_plot, y_plot, 'go', markersize=4, alpha=0.5, zorder=6)

        # Draw labeled emergences (red stars - large)
        for i, (x_label, y_label) in enumerate(labels):
            # labels are stored in ego coordinates (x=forward, y=left)
            # Convert to plot coordinates: x_ego -> y_plot, y_ego (left +) -> x_plot (left -)
            x_plot, y_plot = -y_label, x_label

            # Check if this is selected
            if self.selected_point is not None and self.selected_point == i:
                # Highlight selected point
                ax.plot(x_plot, y_plot, '*', color='yellow', markersize=25,
                       markeredgecolor='orange', markeredgewidth=2, zorder=9)
            else:
                ax.plot(x_plot, y_plot, '*', color='red', markersize=20,
                       markeredgecolor='darkred', markeredgewidth=1.5, zorder=8)

            # Add label number
            ax.text(x_plot, y_plot + 2, f'#{i+1}', fontsize=7, ha='center',
                   color='red', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            alpha=0.8, edgecolor='red'), zorder=8)

        # Add range circles
        for r in [10, 20, 30, 40]:
            circle = Circle((0, 0), r, fill=False, edgecolor='gray',
                          linestyle=':', linewidth=1, alpha=0.4, zorder=1)
            ax.add_patch(circle)
            ax.text(0, r, f'{r}m', fontsize=7, color='gray', alpha=0.6, ha='center')

    def get_ego_pose(self, sample_token):
        """Get ego pose for a sample"""
        sample = self.nusc.get('sample', sample_token)
        sample_data_token = sample['data']['LIDAR_TOP']
        sample_data = self.nusc.get('sample_data', sample_data_token)
        ego_pose = self.nusc.get('ego_pose', sample_data['ego_pose_token'])

        ego_translation = np.array(ego_pose['translation'])
        ego_rotation = Quaternion(ego_pose['rotation'])

        return ego_translation, ego_rotation.rotation_matrix

    def global_to_ego(self, global_position, ego_translation, ego_rotation_matrix):
        """Transform global coordinates to ego-relative coordinates"""
        global_pos = np.array(global_position)
        ego_relative = ego_rotation_matrix.T @ (global_pos - ego_translation)
        return ego_relative

    def on_click(self, event):
        """Handle mouse click on BEV"""
        # Only handle clicks on BEV axes
        if event.inaxes != self.ax_bev:
            return

        if event.xdata is None or event.ydata is None:
            return

        # Click coordinates are in plot space (x=left/right, y=forward)
        x_plot, y_plot = event.xdata, event.ydata

        # Convert to ego coordinates: x_plot=-y_ego, y_plot=x_ego
        # Therefore: x_ego=y_plot, y_ego=-x_plot
        x_ego, y_ego = y_plot, -x_plot

        # Get current labels
        sample_token = self.sample_tokens[self.current_sample_idx]
        if sample_token not in self.labels[self.current_scene_token]:
            self.labels[self.current_scene_token][sample_token] = []

        current_labels = self.labels[self.current_scene_token][sample_token]

        if event.button == 1:  # Left click - add label
            # Check if clicked near existing point (for selection)
            clicked_existing = False
            for i, (lx_ego, ly_ego) in enumerate(current_labels):
                # Convert label to plot coordinates for distance calculation
                lx_plot, ly_plot = -ly_ego, lx_ego
                dist = np.sqrt((x_plot - lx_plot)**2 + (y_plot - ly_plot)**2)
                if dist < 3:  # Within 3 meters
                    self.selected_point = i
                    clicked_existing = True
                    break

            if not clicked_existing:
                # Add new label in ego coordinates (x=forward, y=left)
                current_labels.append((float(x_ego), float(y_ego)))
                self.selected_point = None
                print(f"Added label at ego coords (forward={x_ego:.2f}m, left={y_ego:.2f}m)")

        elif event.button == 3:  # Right click - remove label
            # Find closest point
            if current_labels:
                distances = []
                for lx_ego, ly_ego in current_labels:
                    lx_plot, ly_plot = -ly_ego, lx_ego
                    dist = np.sqrt((x_plot - lx_plot)**2 + (y_plot - ly_plot)**2)
                    distances.append(dist)

                min_idx = np.argmin(distances)

                if distances[min_idx] < 5:  # Within 5 meters
                    removed = current_labels.pop(min_idx)
                    self.selected_point = None
                    print(f"Removed label at ego coords (forward={removed[0]:.2f}m, left={removed[1]:.2f}m)")

        # Update view
        self.update_view()

    def clear_selection(self):
        """Clear selected point"""
        self.selected_point = None
        self.update_view()

    def prev_scene(self):
        """Go to previous scene"""
        if self.current_scene_idx > 0:
            self.current_scene_idx -= 1
            self.load_scene_samples()
            self.update_view()

    def next_scene(self):
        """Go to next scene"""
        if self.current_scene_idx < len(self.scenes) - 1:
            self.current_scene_idx += 1
            self.load_scene_samples()
            self.update_view()

    def prev_sample(self):
        """Go to previous sample (time step)"""
        if self.current_sample_idx > 0:
            self.current_sample_idx -= 1
            self.selected_point = None
            self.update_view()

    def next_sample(self):
        """Go to next sample (time step)"""
        if self.current_sample_idx < len(self.sample_tokens) - 1:
            self.current_sample_idx += 1
            self.selected_point = None
            self.update_view()

    def save_labels(self):
        """Save labels to pickle file"""
        # Create output directory
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Count total labels
        total_labels = sum(len(samples) for scene_samples in self.labels.values()
                          for samples in scene_samples.values())

        # Save
        with open(self.output_path, 'wb') as f:
            pickle.dump(self.labels, f)

        messagebox.showinfo("Save Successful",
                           f"Saved {total_labels} labels to:\n{self.output_path}")
        print(f"Saved labels to {self.output_path}")
        print(f"Total labels: {total_labels}")

    def load_labels(self):
        """Load labels from pickle file"""
        if not self.output_path.exists():
            messagebox.showwarning("File Not Found",
                                  f"No saved labels found at:\n{self.output_path}")
            return

        with open(self.output_path, 'rb') as f:
            self.labels = pickle.load(f)

        # Count total labels
        total_labels = sum(len(samples) for scene_samples in self.labels.values()
                          for samples in scene_samples.values())

        messagebox.showinfo("Load Successful",
                           f"Loaded {total_labels} labels from:\n{self.output_path}")
        print(f"Loaded labels from {self.output_path}")
        print(f"Total labels: {total_labels}")

        # Update view
        self.update_view()

    def quit_app(self):
        """Quit application"""
        if messagebox.askyesno("Quit", "Save labels before quitting?"):
            self.save_labels()

        self.root.quit()
        self.root.destroy()

    def run(self):
        """Start the GUI event loop"""
        print("\n" + "="*80)
        print("EMERGENCE MANUAL LABELING TOOL")
        print("="*80)
        print(f"Dataset: {self.version}")
        print(f"Scenes: {len(self.scenes)}")
        print(f"Output: {self.output_path}")
        print("\nControls:")
        print("  - LEFT CLICK on BEV: Add emergence label")
        print("  - RIGHT CLICK on label: Remove label")
        print("  - LEFT/RIGHT arrow keys: Navigate samples")
        print("  - ESC: Clear selection")
        print("  - Q: Quit")
        print("="*80 + "\n")

        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description='Manual Emergence Labeling Tool')
    parser.add_argument('--dataroot', type=str, required=True,
                       help='Path to nuScenes dataset')
    parser.add_argument('--version', type=str, default='v1.0-mini',
                       help='nuScenes version (default: v1.0-mini)')
    parser.add_argument('--output', type=str, default='data/manual_labels.pkl',
                       help='Output path for labels (default: data/manual_labels.pkl)')

    args = parser.parse_args()

    # Create and run tool
    tool = EmergenceLabelingTool(
        dataroot=args.dataroot,
        version=args.version,
        output_path=args.output
    )

    tool.run()


if __name__ == '__main__':
    main()

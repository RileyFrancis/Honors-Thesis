import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

class BrainViewer3D:
    """Interactive 3D viewer for Gordon ROI brain data"""
    
    def __init__(self, csv_path=None):
        """
        Initialize the brain viewer
        
        Parameters:
        -----------
        csv_path : str, optional
            Path to the Gordon ROI Labels CSV file
        """
        self.df = None
        self.fig = None
        
        self.load_data(csv_path)
    

    def load_data(self, csv_path):
        """
        Load Gordon ROI data from CSV file
        
        Parameters:
        -----------
        csv_path : str
            Path to the CSV file
        """
        self.df = pd.read_csv(csv_path)
        self._process_data()
        print(f"Loaded {len(self.df)} ROIs from {csv_path}")
    
    def _get_group_colors(self):
        """
        Assign a consistent color to each Gordon group
        
        Returns:
        --------
        dict
            Dictionary mapping group names to RGB color strings
        """
        # Define a color palette for the networks
        # Using distinct, visually appealing colors
        color_palette = [
            'rgb(31, 119, 180)',   # Blue
            'rgb(255, 127, 14)',   # Orange
            'rgb(44, 160, 44)',    # Green
            'rgb(214, 39, 40)',    # Red
            'rgb(148, 103, 189)',  # Purple
            'rgb(140, 86, 75)',    # Brown
            'rgb(227, 119, 194)',  # Pink
            'rgb(127, 127, 127)',  # Gray
            'rgb(188, 189, 34)',   # Yellow-green
            'rgb(23, 190, 207)',   # Cyan
            'rgb(174, 199, 232)',  # Light blue
            'rgb(255, 187, 120)',  # Light orange
            'rgb(152, 223, 138)',  # Light green
            'rgb(255, 152, 150)',  # Light red
            'rgb(197, 176, 213)',  # Light purple
        ]
        
        groups = sorted(self.df['gordon_group'].unique())
        group_colors = {}
        
        for i, group in enumerate(groups):
            group_colors[group] = color_palette[i % len(color_palette)]
        
        return group_colors
    
    def _process_data(self):
        """Process and clean the loaded data"""
        # Remove rows with missing coordinates or group
        self.df = self.df.dropna(subset=['x.mni', 'y.mni', 'z.mni', 'gordon_group'])
        
        # Strip whitespace from string columns
        string_cols = ['label', 'gordon_group']
        for col in string_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].str.strip()
        
        # Assign consistent colors per Gordon group
        group_colors = self._get_group_colors()
        self.df['color_rgb'] = self.df['gordon_group'].map(group_colors)
        
        # Create hover text
        self.df['hover_text'] = self.df.apply(
            lambda row: (
                f"<b>{row['label']}</b><br>"
                f"ROI: {row['ROI_i']}<br>"
                f"Network: {row['gordon_group']}<br>"
                f"MNI Coordinates:<br>"
                f"  X: {row['x.mni']:.2f}<br>"
                f"  Y: {row['y.mni']:.2f}<br>"
                f"  Z: {row['z.mni']:.2f}"
            ),
            axis=1
        )
    
    def create_3d_plot(self, show_axes=True, show_grid=True, marker_size=8):
        """
        Create the interactive 3D scatter plot
        
        Parameters:
        -----------
        show_axes : bool, default=True
            Show coordinate axes
        show_grid : bool, default=True
            Show grid lines
        marker_size : int, default=8
            Size of the node markers
        
        Returns:
        --------
        plotly.graph_objects.Figure
            The interactive figure
        """
        if self.df is None:
            raise ValueError("No data loaded. Use load_data() or load_sample_data() first.")
        
        # Create figure
        fig = go.Figure()
        
        # Get unique groups and their colors
        groups = sorted(self.df['gordon_group'].unique())
        group_colors = self._get_group_colors()
        
        # Add a trace for each functional network
        for group in groups:
            group_data = self.df[self.df['gordon_group'] == group]
            
            fig.add_trace(go.Scatter3d(
                x=group_data['x.mni'],
                y=group_data['y.mni'],
                z=group_data['z.mni'],
                mode='markers',
                name=group,
                marker=dict(
                    size=marker_size,
                    color=group_colors[group],  # Single color per group
                    line=dict(color='white', width=0.5),
                    opacity=0.9
                ),
                text=group_data['hover_text'],
                hovertemplate='%{text}<extra></extra>',
                customdata=group_data[['ROI_i', 'label', 'gordon_group']].values
            ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text='<b>Interactive 3D Brain ROI Viewer</b><br>'
                     '<sub>Gordon ROI Parcellation (MNI Coordinates)</sub>',
                x=0.5,
                xanchor='center'
            ),
            scene=dict(
                xaxis=dict(
                    title='X (Left-Right)',
                    showgrid=show_grid,
                    showbackground=True,
                    backgroundcolor='rgb(230, 230,230)',
                    gridcolor='white',
                    showspikes=False
                ),
                yaxis=dict(
                    title='Y (Posterior-Anterior)',
                    showgrid=show_grid,
                    showbackground=True,
                    backgroundcolor='rgb(230, 230, 230)',
                    gridcolor='white',
                    showspikes=False
                ),
                zaxis=dict(
                    title='Z (Inferior-Superior)',
                    showgrid=show_grid,
                    showbackground=True,
                    backgroundcolor='rgb(230, 230, 230)',
                    gridcolor='white',
                    showspikes=False
                ),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                aspectmode='data'
            ),
            showlegend=True,
            legend=dict(
                title=dict(text='<b>Functional Networks</b>'),
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='black',
                borderwidth=1
            ),
            width=1400,
            height=1200,
            hovermode='closest',
            template='plotly_white'
        )
        
        self.fig = fig
        return fig
    
    def show(self, **kwargs):
        """
        Display the 3D plot
        
        Parameters:
        -----------
        **kwargs : dict
            Additional arguments to pass to create_3d_plot()
        """
        if self.fig is None:
            self.create_3d_plot(**kwargs)
        self.fig.show()
    
    def save_html(self, filename='brain_viewer_3d.html', **kwargs):
        """
        Save the interactive plot as an HTML file
        
        Parameters:
        -----------
        filename : str, default='brain_viewer_3d.html'
            Output filename
        **kwargs : dict
            Additional arguments to pass to create_3d_plot()
        """
        if self.fig is None:
            self.create_3d_plot(**kwargs)
        self.fig.write_html(filename)
        print(f"Saved interactive plot to {filename}")
    
    def get_network_summary(self):
        """
        Get summary statistics for each functional network
        
        Returns:
        --------
        pandas.DataFrame
            Summary table with ROI counts per network
        """
        if self.df is None:
            raise ValueError("No data loaded.")
        
        summary = self.df.groupby('gordon_group').agg({
            'ROI_i': 'count',
            'x.mni': ['mean', 'std'],
            'y.mni': ['mean', 'std'],
            'z.mni': ['mean', 'std']
        }).round(2)
        
        summary.columns = ['Count', 'X_mean', 'X_std', 'Y_mean', 'Y_std', 'Z_mean', 'Z_std']
        return summary
    
    def filter_by_network(self, networks):
        """
        Create a viewer with only specific networks
        
        Parameters:
        -----------
        networks : list or str
            Network name(s) to include
        
        Returns:
        --------
        BrainViewer3D
            New viewer instance with filtered data
        """
        if isinstance(networks, str):
            networks = [networks]
        
        filtered_viewer = BrainViewer3D()
        filtered_viewer.df = self.df[self.df['gordon_group'].isin(networks)].copy()
        filtered_viewer._process_data()
        return filtered_viewer
    
    def filter_by_hemisphere(self, hemisphere):
        """
        Create a viewer with only left ('L') or right ('R') hemisphere
        
        Parameters:
        -----------
        hemisphere : str
            'L' for left, 'R' for right
        
        Returns:
        --------
        BrainViewer3D
            New viewer instance with filtered data
        """
        filtered_viewer = BrainViewer3D()
        filtered_viewer.df = self.df[
            self.df['label'].str.contains(f'_{hemisphere}_')
        ].copy()
        filtered_viewer._process_data()
        return filtered_viewer


# Example usage functions
def example_basic():
    """Basic example with sample data"""
    viewer = BrainViewer3D()
    viewer.show()

def example_with_file(csv_path):
    """Example loading from file"""
    viewer = BrainViewer3D(csv_path)
    viewer.show()
    
    # Print summary
    print("\nNetwork Summary:")
    print(viewer.get_network_summary())

def example_filtered():
    """Example with filtered networks"""
    viewer = BrainViewer3D()
    
    # Show only Default and Visual networks
    filtered = viewer.filter_by_network(['Default', 'Visual'])
    filtered.show()

def example_save_html():
    """Example saving to HTML file"""
    viewer = BrainViewer3D()
    viewer.save_html('brain_viewer_3d.html', marker_size=10)


if __name__ == "__main__":
    # Load the full dataset
    print("=== Loading Brain ROI Data ===")
    viewer = BrainViewer3D('/shared/healthinfolab/datasets/ABCD/Irritability/WholeBrainTaskfMRI/UConn/data/Gordon_ROI_Labels.csv')
    
    # Show network summary
    print("\nNetwork Summary:")
    print(viewer.get_network_summary())
    
    # Save to HTML file (better for remote servers)
    print("\nSaving interactive plot to HTML...")
    viewer.save_html('brain_viewer_3d.html')
    print("✓ Done! Download 'brain_viewer_3d.html' and open in your browser.")
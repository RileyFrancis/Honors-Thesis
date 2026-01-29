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
        
        if csv_path:
            self.load_data(csv_path)
        else:
            # Load sample data if no path provided
            self.load_sample_data()
    
    def load_sample_data(self):
        """Load sample Gordon ROI data"""
        sample_data = """ROI_i,label,red,green,blue,alpha,gordon_group,x.mni,y.mni,z.mni,voxel_n
1,1_L_Default,0,0,1,1.000,Default,-11.2,-52.4,36.5,
2,2_L_SMhand,1,0,0,1.000,SMhand,-18.8,-48.7,65,
3,3_L_SMmouth,0,1,0,1.000,SMmouth,-51.8,-7.8,38.5,
4,4_L_Default,0,0,0.173,1.000,Default,-11.7,26.7,57,
5,5_L_Visual,1,0.102,0.725,1.000,Visual,-18.4,-85.5,21.6,
6,6_L_Default,1,0.827,0,1.000,Default,-47.2,-58,30.8,
7,7_L_FrontoParietal,0,0.345,0,1.000,FrontoParietal,-38.1,48.8,10.5,
8,8_L_Visual,0.518,0.518,1,1.000,Visual,-16.8,-60.1,-5.4,
9,9_L_FrontoParietal,0.62,0.31,0.275,1.000,FrontoParietal,-55.9,-47.7,-9.3,
10,10_L_Auditory,0,1,0.757,1.000,Auditory,-32,-29.3,15.6,
11,11_L_None,0,0.518,0.584,1.000,None,-29.3,5.3,-27.4,
12,12_L_CinguloParietal,0,0,0.482,1.000,CinguloParietal,-6.1,-26,28.5,
13,13_L_Visual,0.863,0.078,0.235,1.000,Visual,-10.5,-95.8,0.7,
14,14_L_Auditory,1,0.549,0,1.000,Auditory,-55.7,-35.7,13.7,
15,15_L_SMhand,0.502,0.502,0,1.000,SMhand,-37.5,-26.9,56.4,
16,16_L_CinguloOperc,0,1,1,1.000,CinguloOperc,-38.2,10.7,3.1,
17,17_L_FrontoParietal,1,0.753,0.796,1.000,FrontoParietal,-42.3,34.8,27.9,
18,18_L_Default,0.647,0.165,0.165,1.000,Default,-8.8,-54.6,26.5,
19,19_L_Auditory,0.961,1,0.98,1.000,Auditory,-57.9,-19.8,8.8,
20,20_R_SMhand,0.196,0.804,0.196,1.000,SMhand,23.3,-26.7,67.2"""
        
        from io import StringIO
        self.df = pd.read_csv(StringIO(sample_data))
        self._process_data()
        print(f"Loaded sample data with {len(self.df)} ROIs")
    
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
    
    def _process_data(self):
        """Process and clean the loaded data"""
        # Remove rows with missing coordinates or group
        self.df = self.df.dropna(subset=['x.mni', 'y.mni', 'z.mni', 'gordon_group'])
        
        # Strip whitespace from string columns
        string_cols = ['label', 'gordon_group']
        for col in string_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].str.strip()
        
        # Create RGB color strings for plotly
        self.df['color_rgb'] = self.df.apply(
            lambda row: f"rgb({int(row['red']*255)}, {int(row['green']*255)}, {int(row['blue']*255)})",
            axis=1
        )
        
        # Create hover text
        self.df['hover_text'] = self.df.apply(
            lambda row: (
                f"<b>{row['label']}</b><br>"
                f"ROI: {row['ROI_i']}<br>"
                f"Network: {row['gordon_group']}<br>"
                f"MNI Coordinates:<br>"
                f"  X: {row['x.mni']:.2f}<br>"
                f"  Y: {row['y.mni']:.2f}<br>"
                f"  Z: {row['z.mni']:.2f}<br>"
                f"RGB: ({int(row['red']*255)}, {int(row['green']*255)}, {int(row['blue']*255)})"
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
        
        # Get unique groups
        groups = self.df['gordon_group'].unique()
        
        # Add a trace for each functional network
        for group in sorted(groups):
            group_data = self.df[self.df['gordon_group'] == group]
            
            fig.add_trace(go.Scatter3d(
                x=group_data['x.mni'],
                y=group_data['y.mni'],
                z=group_data['z.mni'],
                mode='markers',
                name=group,
                marker=dict(
                    size=marker_size,
                    color=group_data['color_rgb'],
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
                     '<sub>Gordon Parcellation - MNI Coordinates</sub>',
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
            height=900,
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
    # Example 1: Load sample data and save to HTML (for remote servers)
    print("=== Example 1: Basic Viewer ===")
    viewer = BrainViewer3D()
    
    # Show network summary
    print("\nNetwork Summary:")
    print(viewer.get_network_summary())
    
    # Save to HTML file (better for remote servers)
    print("\nSaving interactive plot to HTML...")
    viewer.save_html('brain_viewer_3d.html')
    print("✓ Done! Download 'brain_viewer_3d.html' and open in your browser.")
    
    # If you want to load your actual data file:
    print("\n=== Loading your data file ===")
    try:
        viewer_full = BrainViewer3D('/shared/healthinfolab/datasets/ABCD/Irritability/WholeBrainTaskfMRI/UConn/data/Gordon_ROI_Labels.csv')
        print("\nFull Dataset Network Summary:")
        print(viewer_full.get_network_summary())
        viewer_full.save_html('brain_viewer_full_data.html')
        print("\n✓ Saved full dataset to 'brain_viewer_full_data.html'")
    except FileNotFoundError:
        print("\nNote: Full data file not found. Using sample data only.")
    except Exception as e:
        print(f"\nNote: Could not load full data file: {e}")
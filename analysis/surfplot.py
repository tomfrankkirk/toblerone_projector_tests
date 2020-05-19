"""Basic mesh plotting function. Tom Kirk, Jan 2020"""

import plotly.graph_objects as go 
import plotly

def surf_fig(surf, data, cmap=None):
    """
    Plot metric data onto surface defined by points, tris arrays. 
    Uses plotly.Mesh3d, see that documentation for more information. 
    Default color map: [ 0 -> gold, 0.5 -> mediumtuquoise, 1-> magenta ]
    
    Args: 
        points: px3 array of surface nodes
        tris: tx3 array of triangle indices into points matrix 
        data: px1 array of data to plot 
        cmap: colour spec for plotly (see default example above)
        
    Returns: 
        plotly.Figure object
    """
    
    return go.Figure(data=[surf_mesh(surf, data, cmap)])

def surf_mesh(surf, data, cmap=None):
    assert data.size == surf.points.shape[0], 'data does not match surface size'
    x, y, z = surf.points.T
    i, j, k = surf.tris.T
    
    if cmap is None: 
        cmap = [[0, 'gold'],[0.5, 'mediumturquoise'],[1, 'magenta']]
        
    return go.Mesh3d(
            x=x,y=y,z=z,i=i,j=j,k=k,
            colorscale=cmap,
            # Intensity of each vertex, which will be interpolated and color-coded
            intensity=data,
            showscale=True
        )
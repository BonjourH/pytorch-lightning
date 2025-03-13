import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
import math

def fisheye_distortion(r, k1, k2, k3, k4):
    """
    Compute the fisheye distortion factor for a given radius r.
    
    Args:
        r: Undistorted radius (can be numpy array or torch tensor)
        k1, k2, k3, k4: Distortion coefficients
        
    Returns:
        Distortion scaling factor
    """
    theta = np.arctan(r)
    theta2 = theta * theta
    theta4 = theta2 * theta2
    theta6 = theta4 * theta2
    theta8 = theta4 * theta4
    
    theta_d = theta * (1 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8)
    scaling = np.divide(theta_d, r, out=np.ones_like(theta_d), where=r>0)
    
    return scaling

def barrel_distortion(r, k1, k2, k3):
    """
    Compute the barrel/pincushion distortion factor for a given radius r.
    
    Args:
        r: Undistorted radius (can be numpy array or torch tensor)
        k1, k2, k3: Distortion coefficients
        
    Returns:
        Distortion factor
    """
    r2 = r * r
    r4 = r2 * r2
    r6 = r4 * r2
    
    return 1 + k1 * r2 + k2 * r4 + k3 * r6

def visualize_distortions():
    # Set up the figure
    plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, height_ratios=[1, 1])
    
    # Create radii values from 0 to 3
    radii = np.linspace(0, 3, 1000)
    angles_degrees = np.arctan(radii) * 180 / np.pi  # Convert to degrees for plot
    
    # Fisheye distortion parameters to test
    fisheye_params = [
        {'k1': 0.1, 'k2': 0.0, 'k3': 0.0, 'k4': 0.0, 'label': 'k1=0.1'},
        {'k1': -0.1, 'k2': 0.0, 'k3': 0.0, 'k4': 0.0, 'label': 'k1=-0.1'},
        {'k1': 0.1, 'k2': 0.05, 'k3': 0.0, 'k4': 0.0, 'label': 'k1=0.1, k2=0.05'},
        {'k1': 0.0, 'k2': 0.0, 'k3': 0.01, 'k4': 0.0, 'label': 'k3=0.01'},
        {'k1': 0.1, 'k2': 0.05, 'k3': 0.01, 'k4': 0.005, 'label': 'All positive'}
    ]
    
    # Barrel distortion parameters to test
    barrel_params = [
        {'k1': 0.1, 'k2': 0.0, 'k3': 0.0, 'label': 'k1=0.1 (barrel)'},
        {'k1': -0.1, 'k2': 0.0, 'k3': 0.0, 'label': 'k1=-0.1 (pincushion)'},
        {'k1': 0.1, 'k2': 0.05, 'k3': 0.0, 'label': 'k1=0.1, k2=0.05'},
        {'k1': 0.0, 'k2': 0.0, 'k3': 0.01, 'label': 'k3=0.01'},
        {'k1': -0.1, 'k2': 0.05, 'k3': -0.01, 'label': 'Mixed'}
    ]
    
    # Plot fisheye distortion vs angle
    ax1 = plt.subplot(gs[0, 0])
    for params in fisheye_params:
        distortion = fisheye_distortion(
            radii, params['k1'], params['k2'], params['k3'], params['k4']
        )
        ax1.plot(angles_degrees, distortion, label=params['label'])
    
    ax1.set_title('Fisheye Distortion Scaling vs Angle')
    ax1.set_xlabel('Angle (degrees)')
    ax1.set_ylabel('Distortion Scale Factor')
    ax1.grid(True)
    ax1.legend()
    ax1.set_xlim(0, 70)  # Most fisheye lenses have around 180 degree FOV (so 90 degrees from center)
    
    # Plot barrel distortion vs radius
    ax2 = plt.subplot(gs[0, 1])
    for params in barrel_params:
        distortion = barrel_distortion(
            radii, params['k1'], params['k2'], params['k3']
        )
        ax2.plot(radii, distortion, label=params['label'])
    
    ax2.set_title('Barrel/Pincushion Distortion Factor vs Radius')
    ax2.set_xlabel('Normalized Radius')
    ax2.set_ylabel('Distortion Factor')
    ax2.grid(True)
    ax2.legend()
    ax2.set_xlim(0, 1.5)  # Typically normalized coordinates are within this range
    
    # Visual grid distortion for fisheye
    ax3 = plt.subplot(gs[1, 0])
    visualize_grid_distortion(ax3, fisheye_distortion, 
                              fisheye_params[4]['k1'], 
                              fisheye_params[4]['k2'], 
                              fisheye_params[4]['k3'], 
                              fisheye_params[4]['k4'], 
                              "Fisheye Distortion")
    
    # Visual grid distortion for barrel
    ax4 = plt.subplot(gs[1, 1])
    visualize_grid_distortion(ax4, barrel_distortion, 
                              barrel_params[0]['k1'], 
                              barrel_params[0]['k2'], 
                              barrel_params[0]['k3'], 
                              0.0,  # k4 not used for barrel
                              "Barrel Distortion")
    
    plt.tight_layout()
    plt.savefig('camera_distortion_comparison.png', dpi=300)
    plt.show()

def visualize_grid_distortion(ax, distortion_func, k1, k2, k3, k4, title):
    """
    Create a visual grid to demonstrate distortion effect.
    """
    # Create a grid of points
    n = 10  # number of grid lines in each direction
    limit = 1.0  # limit of the grid
    
    x = np.linspace(-limit, limit, n)
    y = np.linspace(-limit, limit, n)
    
    # Create mesh grid
    X, Y = np.meshgrid(x, y)
    
    # Calculate radius for each point
    R = np.sqrt(X**2 + Y**2)
    
    # Get distortion factor
    if distortion_func == fisheye_distortion:
        distortion = distortion_func(R, k1, k2, k3, k4)
    else:  # barrel distortion
        distortion = distortion_func(R, k1, k2, k3)
    
    # Apply distortion
    X_distorted = X * distortion
    Y_distorted = Y * distortion
    
    # Plot original grid
    ax.plot(X, Y, 'k-', alpha=0.3)
    ax.plot(X.T, Y.T, 'k-', alpha=0.3)
    
    # Plot distorted grid
    ax.plot(X_distorted, Y_distorted, 'r-')
    ax.plot(X_distorted.T, Y_distorted.T, 'r-')
    
    # Set limits and title
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(False)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

def generate_distortion_visualization_3d():
    """
    Generate 3D visualizations of the distortion effects
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    # Create a new figure
    fig = plt.figure(figsize=(15, 7))
    
    # Create grid for 3D plots
    x = np.linspace(-1, 1, 50)
    y = np.linspace(-1, 1, 50)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    
    # Fisheye distortion with parameters
    k1, k2, k3, k4 = 0.1, 0.05, 0.01, 0.005
    Z_fisheye = fisheye_distortion(R, k1, k2, k3, k4)
    
    # Barrel distortion with parameters
    k1, k2, k3 = 0.1, 0.05, 0.01
    Z_barrel = barrel_distortion(R, k1, k2, k3)
    
    # Plot fisheye distortion surface
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(X, Y, Z_fisheye, cmap='viridis')
    ax1.set_title('Fisheye Distortion Factor')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Distortion Factor')
    
    # Plot barrel distortion surface
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(X, Y, Z_barrel, cmap='plasma')
    ax2.set_title('Barrel Distortion Factor')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Distortion Factor')
    
    plt.tight_layout()
    plt.savefig('3d_distortion_visualization.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    visualize_distortions()
    generate_distortion_visualization_3d()
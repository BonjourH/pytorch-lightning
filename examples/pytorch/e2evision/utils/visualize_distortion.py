import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import argparse

def fisheye_distortion(r, k1, k2, k3, k4, scaled_radius=False):
    """
    Compute the fisheye distortion factor for a given radius r.
    
    Args:
        r: Undistorted radius (can be numpy array)
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
    
    return scaling * r if scaled_radius else scaling

def barrel_distortion(r, k1, k2, k3, scaled_radius=False):
    """
    Compute the barrel/pincushion distortion factor for a given radius r.
    
    Args:
        r: Undistorted radius (can be numpy array)
        k1, k2, k3: Distortion coefficients
        
    Returns:
        Distortion factor
    """
    r2 = r * r
    r4 = r2 * r2
    r6 = r4 * r2
    barrel_distortion = 1 + k1 * r2 + k2 * r4 + k3 * r6
    
    # barrel_distortion = 1 + 3 * k1 * r2 + 5 * k2 * r4 + 7 * k3 * r6, derivation
    return barrel_distortion * r if scaled_radius else barrel_distortion

def visualize_grid_distortion(ax, distortion_func, k1, k2, k3, k4=0.0, title="Distortion", plot_radius=True):
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

def visualize_distortions(fisheye_params, barrel_params, output_file=None, plot_radius=True):
    """
    Create visualizations comparing fisheye and barrel distortion.
    
    Args:
        fisheye_params: List of dictionaries with k1, k2, k3, k4, label keys
        barrel_params: List of dictionaries with k1, k2, k3, label keys
        output_file: Optional filename to save the plot
    """
    # Set up the figure
    plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, height_ratios=[1, 1])
    
    # Create radii values from 0 to 3
    radii = np.linspace(0, 10, 1000)
    angles_degrees = np.arctan(radii) * 180 / np.pi  # Convert to degrees for plot
    
    # Plot fisheye distortion vs angle
    ax1 = plt.subplot(gs[0, 0])
    for params in fisheye_params:
        distortion = fisheye_distortion(
            radii, params['k1'], params['k2'], params['k3'], params['k4'], plot_radius
        )
        ax1.plot(angles_degrees, distortion, label=params['label'])
    
    ax1.set_title(f'Fisheye Distortion Scaling vs Angle')
    if plot_radius:
        ax1.set_title('Fisheye Distorted Radius vs Angle')
    ax1.set_xlabel('Angle (degrees)')
    ax1.set_ylabel('Distortion Scale Factor')
    ax1.grid(True)
    ax1.legend()
    ax1.set_xlim(0, 89)  # Most fisheye lenses have around 180 degree FOV (so 90 degrees from center)
    
    # Plot barrel distortion vs radius
    normalized_r = 1.5
    radii = np.linspace(0, normalized_r, 1000)
    ax2 = plt.subplot(gs[0, 1])
    for params in barrel_params:
        distortion = barrel_distortion(
            radii, params['k1'], params['k2'], params['k3'], plot_radius
        )
        ax2.plot(radii, distortion, label=params['label'])
    
    ax2.set_title('Barrel/Pincushion Distortion Factor vs Radius')
    if plot_radius:
        ax2.set_title('Barrel/Pincushion Distorted Radius vs Radius')
    ax2.set_xlabel('Normalized Radius')
    ax2.set_ylabel('Distortion Factor')
    ax2.grid(True)
    ax2.legend()
    ax2.set_xlim(0, normalized_r)  # Typically normalized coordinates are within this range
    ax2.set_ylim(0, normalized_r)
    
    # Use the first set of parameters for the grid visualization
    if fisheye_params:
        # Visual grid distortion for fisheye
        ax3 = plt.subplot(gs[1, 0])
        visualize_grid_distortion(ax3, fisheye_distortion, 
                                fisheye_params[0]['k1'], 
                                fisheye_params[0]['k2'], 
                                fisheye_params[0]['k3'], 
                                fisheye_params[0]['k4'], 
                                f"Fisheye Distortion: {fisheye_params[0]['label']}")
    
    if barrel_params:
        # Visual grid distortion for barrel
        ax4 = plt.subplot(gs[1, 1])
        visualize_grid_distortion(ax4, barrel_distortion, 
                                barrel_params[0]['k1'], 
                                barrel_params[0]['k2'], 
                                barrel_params[0]['k3'], 
                                0.0,  # k4 not used for barrel
                                f"Barrel Distortion: {barrel_params[0]['label']}")
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300)
    
    plt.show()

def generate_distortion_visualization_3d(fisheye_k1, fisheye_k2, fisheye_k3, fisheye_k4, 
                                        barrel_k1, barrel_k2, barrel_k3, 
                                        output_file=None):
    """
    Generate 3D visualizations of the distortion effects
    
    Args:
        fisheye_k1, fisheye_k2, fisheye_k3, fisheye_k4: Fisheye distortion coefficients
        barrel_k1, barrel_k2, barrel_k3: Barrel distortion coefficients
        output_file: Optional filename to save the plot
    """
    # Create a new figure
    fig = plt.figure(figsize=(15, 7))
    
    # Create grid for 3D plots
    x = np.linspace(-1, 1, 50)
    y = np.linspace(-1, 1, 50)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    
    # Fisheye distortion
    Z_fisheye = fisheye_distortion(R, fisheye_k1, fisheye_k2, fisheye_k3, fisheye_k4)
    
    # Barrel distortion
    Z_barrel = barrel_distortion(R, barrel_k1, barrel_k2, barrel_k3)
    
    # Plot fisheye distortion surface
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(X, Y, Z_fisheye, cmap='viridis')
    ax1.set_title(f'Fisheye Distortion: k1={fisheye_k1}, k2={fisheye_k2}, k3={fisheye_k3}, k4={fisheye_k4}')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Distortion Factor')
    
    # Plot barrel distortion surface
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(X, Y, Z_barrel, cmap='plasma')
    ax2.set_title(f'Barrel Distortion: k1={barrel_k1}, k2={barrel_k2}, k3={barrel_k3}')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Distortion Factor')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300)
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize camera distortion models')
    
    # Fisheye parameters
    parser.add_argument('--fisheye_k1', type=float, default=-4.1464596584104653e-02, help='Fisheye k1 coefficient')
    parser.add_argument('--fisheye_k2', type=float, default=9.0031786097134383e-03, help='Fisheye k2 coefficient')
    parser.add_argument('--fisheye_k3', type=float, default=-1.3618604927618041e-02, help='Fisheye k3 coefficient')
    parser.add_argument('--fisheye_k4', type=float, default=3.7757395282444445e-03, help='Fisheye k4 coefficient')
    
    # Barrel parameters
    parser.add_argument('--barrel_k1', type=float, default=-6.2441600000000008e-01, help='Barrel k1 coefficient')
    parser.add_argument('--barrel_k2', type=float, default=3.0044600000000005e-01, help='Barrel k2 coefficient')
    parser.add_argument('--barrel_k3', type=float, default=0.0, help='Barrel k3 coefficient')
    
    # Output files
    parser.add_argument('--output_2d', type=str, default=None, help='Output file for 2D plots')
    parser.add_argument('--output_3d', type=str, default=None, help='Output file for 3D plots')
    
    # Compare multiple parameter sets
    parser.add_argument('--compare', type=bool, default=False, help='Compare with standard parameter sets')
    
    parser.add_argument('--plot_radius', type=bool, default=True, help='Plot distortion')
    
    args = parser.parse_args()
    
    # Create parameter dictionaries for the custom values
    custom_fisheye = {
        'k1': args.fisheye_k1,
        'k2': args.fisheye_k2,
        'k3': args.fisheye_k3,
        'k4': args.fisheye_k4,
        'label': f'Custom (k1={args.fisheye_k1:.3f}, k2={args.fisheye_k2:.3f}, k3={args.fisheye_k3:.3f}, k4={args.fisheye_k4:.3f})'
    }
    
    custom_barrel = {
        'k1': args.barrel_k1,
        'k2': args.barrel_k2,
        'k3': args.barrel_k3,
        'label': f'Custom (k1={args.barrel_k1:.3f}, k2={args.barrel_k2:.3f}, k3={args.barrel_k3:.3f})'
    }
    
    if args.compare:
        # Add standard parameter sets for comparison
        fisheye_params = [
            custom_fisheye,
            # {'k1': 0.1, 'k2': 0.0, 'k3': 0.0, 'k4': 0.0, 'label': 'k1=0.1'},
            # {'k1': 0.0, 'k2': 0.05, 'k3': 0.0, 'k4': 0.0, 'label': 'k2=0.05'},
            # {'k1': 0.0, 'k2': 0.0, 'k3': 0.01, 'k4': 0.0, 'label': 'k3=0.01'},
            # {'k1': 0.0, 'k2': 0.0, 'k3': 0.0, 'k4': 0.005, 'label': 'k4=0.005'}
        ]
        
        barrel_params = [
            custom_barrel,
            {'k1': -3.7552558150588855e-01, 'k2': -1.0846276902696257e+00, 'k3': 0.0, 'label': 'front right'},
            # {'k1': -0.1, 'k2': 0.0, 'k3': 0.0, 'label': 'k1=-0.1 (pincushion)'},
            # {'k1': 0.0, 'k2': 0.05, 'k3': 0.0, 'label': 'k2=0.05'},
            # {'k1': 0.0, 'k2': 0.0, 'k3': 0.01, 'label': 'k3=0.01'}
        ]
    else:
        # Just use the custom parameters
        fisheye_params = [custom_fisheye]
        barrel_params = [custom_barrel]
    
    # Generate the visualizations
    visualize_distortions(fisheye_params, barrel_params, args.output_2d)
    
    generate_distortion_visualization_3d(
        args.fisheye_k1, args.fisheye_k2, args.fisheye_k3, args.fisheye_k4,
        args.barrel_k1, args.barrel_k2, args.barrel_k3,
        args.output_3d
    )

if __name__ == "__main__":
    main()
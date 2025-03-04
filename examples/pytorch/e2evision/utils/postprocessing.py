import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.interpolate import interp1d
from scipy.spatial import KDTree
from base import ObstacleTrajectory, Point3DAccMotion

class TrajectoryOptimizer:
    """Optimize and refine predicted trajectories."""
    
    def __init__(self,
                 smoothing_window: int = 5,
                 min_velocity: float = 0.1,
                 max_acceleration: float = 5.0,
                 max_yaw_rate: float = np.pi/2):
        self.smoothing_window = smoothing_window
        self.min_velocity = min_velocity
        self.max_acceleration = max_acceleration
        self.max_yaw_rate = max_yaw_rate
    
    def optimize_trajectories(self,
                            trajectories: List[ObstacleTrajectory],
                            dt: float = 0.1) -> List[ObstacleTrajectory]:
        """Apply trajectory optimization.
        
        Args:
            trajectories: List of predicted trajectories
            dt: Time step between frames
            
        Returns:
            List of optimized trajectories
        """
        # Apply smoothing
        smoothed_trajs = self.smooth_trajectories(trajectories)
        
        # Apply physical constraints
        constrained_trajs = self.apply_physical_constraints(smoothed_trajs, dt)
        
        # Resolve collisions
        final_trajs = self.resolve_collisions(constrained_trajs)
        
        return final_trajs
    
    def smooth_trajectories(self,
                          trajectories: List[ObstacleTrajectory]) -> List[ObstacleTrajectory]:
        """Apply smoothing to trajectories."""
        smoothed_trajs = []
        
        for traj in trajectories:
            # Create smoothed motion
            smoothed_motion = Point3DAccMotion(
                x=traj.motion.x,
                y=traj.motion.y,
                z=traj.motion.z,
                vx=traj.motion.vx,
                vy=traj.motion.vy,
                vz=traj.motion.vz,
                ax=traj.motion.ax,
                ay=traj.motion.ay,
                az=traj.motion.az
            )
            
            # Apply Gaussian smoothing to velocities and accelerations
            kernel = self._gaussian_kernel(self.smoothing_window)
            
            vel = np.array([traj.motion.vx, traj.motion.vy])
            acc = np.array([traj.motion.ax, traj.motion.ay])
            
            smoothed_vel = np.convolve(vel, kernel, mode='same')
            smoothed_acc = np.convolve(acc, kernel, mode='same')
            
            smoothed_motion.vx = float(smoothed_vel[0])
            smoothed_motion.vy = float(smoothed_vel[1])
            smoothed_motion.ax = float(smoothed_acc[0])
            smoothed_motion.ay = float(smoothed_acc[1])
            
            # Create smoothed trajectory
            smoothed_traj = ObstacleTrajectory(
                id=traj.id,
                motion=smoothed_motion,
                yaw=traj.yaw,
                length=traj.length,
                width=traj.width,
                height=traj.height,
                object_type=traj.object_type,
                t0=traj.t0,
                static=traj.static,
                valid=traj.valid
            )
            
            smoothed_trajs.append(smoothed_traj)
        
        return smoothed_trajs
    
    def apply_physical_constraints(self,
                                 trajectories: List[ObstacleTrajectory],
                                 dt: float) -> List[ObstacleTrajectory]:
        """Apply physical constraints to trajectories."""
        constrained_trajs = []
        
        for traj in trajectories:
            # Check if static
            speed = np.linalg.norm([traj.motion.vx, traj.motion.vy])
            if speed < self.min_velocity:
                traj.static = True
                traj.motion.vx = 0.0
                traj.motion.vy = 0.0
                traj.motion.ax = 0.0
                traj.motion.ay = 0.0
            
            # Limit acceleration
            acc = np.linalg.norm([traj.motion.ax, traj.motion.ay])
            if acc > self.max_acceleration:
                scale = self.max_acceleration / acc
                traj.motion.ax *= scale
                traj.motion.ay *= scale
            
            # Limit yaw rate
            if not traj.static:
                yaw_rate = abs(np.arctan2(traj.motion.vy, traj.motion.vx) - traj.yaw) / dt
                if yaw_rate > self.max_yaw_rate:
                    # Adjust velocity direction to satisfy max yaw rate
                    target_yaw = traj.yaw + np.sign(yaw_rate) * self.max_yaw_rate * dt
                    speed = np.linalg.norm([traj.motion.vx, traj.motion.vy])
                    traj.motion.vx = speed * np.cos(target_yaw)
                    traj.motion.vy = speed * np.sin(target_yaw)
            
            constrained_trajs.append(traj)
        
        return constrained_trajs
    
    def resolve_collisions(self,
                         trajectories: List[ObstacleTrajectory]) -> List[ObstacleTrajectory]:
        """Resolve collisions between trajectories."""
        if len(trajectories) < 2:
            return trajectories
        
        resolved_trajs = trajectories.copy()
        
        # Build KD-tree for efficient neighbor search
        positions = np.array([traj.position for traj in trajectories])
        tree = KDTree(positions)
        
        # Find potential collisions
        pairs = tree.query_pairs(r=5.0)  # 5.0m radius for collision check
        
        for i, j in pairs:
            traj1 = resolved_trajs[i]
            traj2 = resolved_trajs[j]
            
            # Skip if either is static
            if traj1.static or traj2.static:
                continue
            
            # Check collision
            if self._check_collision(traj1, traj2):
                # Resolve by adjusting velocities
                self._resolve_pair_collision(traj1, traj2)
        
        return resolved_trajs
    
    def _gaussian_kernel(self, window_size: int) -> np.ndarray:
        """Create Gaussian smoothing kernel."""
        x = np.linspace(-2, 2, window_size)
        kernel = np.exp(-x**2)
        return kernel / kernel.sum()
    
    def _check_collision(self,
                        traj1: ObstacleTrajectory,
                        traj2: ObstacleTrajectory) -> bool:
        """Check if two trajectories collide."""
        # Simple distance-based collision check
        distance = np.linalg.norm(traj1.position - traj2.position)
        min_distance = max(
            np.hypot(traj1.length, traj1.width),
            np.hypot(traj2.length, traj2.width)
        ) / 2
        
        return distance < min_distance
    
    def _resolve_pair_collision(self,
                              traj1: ObstacleTrajectory,
                              traj2: ObstacleTrajectory):
        """Resolve collision between two trajectories."""
        # Simple velocity-based resolution
        rel_pos = traj2.position - traj1.position
        distance = np.linalg.norm(rel_pos)
        direction = rel_pos / distance
        
        # Adjust velocities in opposite directions
        speed1 = np.linalg.norm([traj1.motion.vx, traj1.motion.vy])
        speed2 = np.linalg.norm([traj2.motion.vx, traj2.motion.vy])
        
        traj1.motion.vx -= direction[0] * speed1 * 0.5
        traj1.motion.vy -= direction[1] * speed1 * 0.5
        traj2.motion.vx += direction[0] * speed2 * 0.5
        traj2.motion.vy += direction[1] * speed2 * 0.5

class TrajectoryInterpolator:
    """Interpolate and extrapolate trajectories."""
    
    def __init__(self,
                 max_extrapolation_time: float = 3.0):
        self.max_extrapolation_time = max_extrapolation_time
    
    def interpolate(self,
                   trajectory: ObstacleTrajectory,
                   timestamps: np.ndarray) -> List[np.ndarray]:
        """Interpolate trajectory at given timestamps.
        
        Args:
            trajectory: Trajectory to interpolate
            timestamps: Array of timestamps for interpolation
            
        Returns:
            List of interpolated positions
        """
        # Get reference time
        t0 = trajectory.t0
        
        # Calculate relative times
        times = timestamps - t0
        
        # Get positions at each timestamp
        positions = []
        for t in times:
            if t < 0:  # Past
                pos = trajectory.position  # Use current position
            elif t > self.max_extrapolation_time:  # Too far in future
                pos = trajectory.position_at(self.max_extrapolation_time)
            else:  # Within valid range
                pos = trajectory.position_at(t)
            positions.append(pos)
        
        return positions
    
    def extrapolate(self,
                    trajectory: ObstacleTrajectory,
                    dt: float,
                    num_steps: int) -> List[np.ndarray]:
        """Extrapolate trajectory into future.
        
        Args:
            trajectory: Trajectory to extrapolate
            dt: Time step
            num_steps: Number of steps to extrapolate
            
        Returns:
            List of extrapolated positions
        """
        times = np.arange(1, num_steps + 1) * dt
        return self.interpolate(trajectory, times + trajectory.t0)

class TrajectoryValidator:
    """Validate trajectory predictions."""
    
    def __init__(self,
                 max_speed: float = 40.0,  # m/s
                 max_acceleration: float = 10.0,  # m/s^2
                 min_size: Tuple[float, float, float] = (0.5, 0.5, 0.5),  # meters
                 max_size: Tuple[float, float, float] = (20.0, 5.0, 5.0)):  # meters
        self.max_speed = max_speed
        self.max_acceleration = max_acceleration
        self.min_size = min_size
        self.max_size = max_size
    
    def validate_trajectory(self, trajectory: ObstacleTrajectory) -> bool:
        """Check if trajectory is physically plausible.
        
        Args:
            trajectory: Trajectory to validate
            
        Returns:
            True if trajectory is valid
        """
        # Check speed
        speed = np.linalg.norm([trajectory.motion.vx, trajectory.motion.vy])
        if speed > self.max_speed:
            return False
        
        # Check acceleration
        acceleration = np.linalg.norm([trajectory.motion.ax, trajectory.motion.ay])
        if acceleration > self.max_acceleration:
            return False
        
        # Check dimensions
        if not (self.min_size[0] <= trajectory.length <= self.max_size[0] and
                self.min_size[1] <= trajectory.width <= self.max_size[1] and
                self.min_size[2] <= trajectory.height <= self.max_size[2]):
            return False
        
        return True
    
    def filter_trajectories(self,
                          trajectories: List[ObstacleTrajectory]) -> List[ObstacleTrajectory]:
        """Filter out invalid trajectories.
        
        Args:
            trajectories: List of trajectories to validate
            
        Returns:
            List of valid trajectories
        """
        return [traj for traj in trajectories if self.validate_trajectory(traj)] 
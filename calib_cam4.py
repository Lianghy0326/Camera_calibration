import numpy as np
import cv2
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
import argparse
import pickle
import json
import pandas as pd
from PIL import Image
from tqdm import tqdm
from scipy.optimize import minimize
from joblib import Parallel, delayed
from itertools import product


class CameraCalibrator:
    """Camera calibration class for estimating intrinsic and extrinsic parameters."""
    
    def __init__(self, config=None):
        """
        Initialize camera calibrator with optional configuration.
        
        Args:
            config (dict, optional): Configuration dictionary with calibration parameters.
        """
        self.config = config or {}
        self.camera_matrix = None
        self.dist_coeffs = None
        self.extrinsic_matrix = None
        self.image_width = 1920
        self.image_height = 1200
        self.image_center = (self.image_width // 2, self.image_height // 2)
        self.image_points = np.zeros((18, 2), dtype=np.float32)
        
    def load_data(self, court_points_path, measurements_path, camera_id, ball_3d_path=None, ball_2d_path=None):
        """
        Load calibration data from files.
        
        Args:
            court_points_path (str): Path to pickle file with court points.
            measurements_path (str): Path to JSON file with physical measurements.
            camera_id (str): Camera ID.
            ball_3d_path (str, optional): Path to 3D ball positions CSV.
            ball_2d_path (str, optional): Path to 2D ball detections CSV.
            
        Returns:
            bool: Success flag
        """
        try:
            # Load court points
            court_data = pickle.load(open(court_points_path, 'rb'))[0] #  {0: (1255, 418), 1: (1430, 888), 2: (569, 452)...
            for i in range(18):
                for i in court_data:
                    # Each value in court is a tuple (x,y)
                    x, y = court_data[i]
                    self.image_points[i,0] = x
                    self.image_points[i,1] = y
            
            
            # Load measurement data
            with open(measurements_path, 'r') as f:
                data = json.load(f)
                camera_measurements = data['camera_id'][camera_id]
                self.vertical_distance = -camera_measurements[0] + 0.5  # Adjust for camera height
                self.measure_distances = camera_measurements[1:]  # 4 point distances
            
            # Define 3D court model based on known dimensions
            self.setup_3d_court_model()
            
            # Load ball data if provided
            if ball_3d_path and ball_2d_path:
                self.ball3d, self.ball2d = self.load_ball_data(ball_3d_path, ball_2d_path)
                print(f"Loaded ball data: {len(self.ball3d)} points")
            else:
                self.ball3d = np.array([])
                self.ball2d = np.array([])
                
            # Set up reference points for distance measurements (對應量測的4個點)
            self.measure_points_3d = np.array([
                [-2.59, 0.5, -5.94],
                [-2.59, 0.5, -1.98],
                [2.59, 0.5, -5.94],
                [2.59, 0.5, -1.98],
            ], dtype=np.float32)
            
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def setup_3d_court_model(self):
        """Define the 3D court model based on standard volleyball court dimensions."""
        # Combined court model for all cameras
        self.object_points = np.array([
            # Left side 9 points
            [-1.98, -2.59, -0.5],
            [-5.94, -2.59, -0.5],
            [-1.98,  2.59, -0.5],
            [-5.94,  2.59, -0.5],
            [-1.98, 0, -0.5],
            [-5.94, 0, -0.5],
            [-6.7, -2.59, -0.5],
            [-6.7, 0, -0.5],
            [-6.7, 2.59, -0.5],
            # Right side 9 points
            [-5.94, 2.59, -0.5],
            [-1.98, 2.59, -0.5],
            [-5.94,-2.59, -0.5],
            [-1.98,-2.59, -0.5],
            [-5.94, 0, -0.5],
            [-1.98, 0, -0.5],
            [-6.7,2.59, -0.5],
            [-6.7, 0, -0.5],
            [-6.7, -2.59, -0.5]
        ], dtype=np.float32)
        
        # Apply rotations based on camera position
        # Note: This transforms the model from volleyball-centric coordinates to camera-centered coordinates
        R1 = np.array([
            [0, 1, 0],
            [0, 0, -1],
            [-1, 0, 0]
        ], dtype=np.float32)
        
        R2 = np.array([
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0]
        ], dtype=np.float32)
        
        # Apply rotations to the two halves of the court
        self.object_points[:9,:] = self.object_points[:9,:] @ R1.T
        self.object_points[9:,:] = self.object_points[9:,:] @ R2.T
            
    def load_ball_data(self, ball3d_path, ball_path):
        """
        Load and synchronize 3D and 2D ball position data.
        
        Args:
            ball3d_path (str): Path to 3D ball positions CSV.
            ball_path (str): Path to 2D ball detections CSV.
            
        Returns:
            tuple: (ball3d, ball2d) arrays with matching frames
        """
        ball3d_df = pd.read_csv(ball3d_path)
        ball_df = pd.read_csv(ball_path)
        
        # Extract frames common to both datasets
        common_frames = np.intersect1d(ball3d_df['Frame'].values, ball_df['Frame'].values)
        
        # Filter data to common frames
        ball3d_df = ball3d_df[ball3d_df['Frame'].isin(common_frames)]
        ball_df = ball_df[ball_df['Frame'].isin(common_frames)]
        
        # Extract position data
        ball3d = ball3d_df[['X', 'Y', 'Z']].values.astype(np.float32)
        ball2d = ball_df[['X', 'Y']].values.astype(np.float32)
        
        # Adjust 3D coordinates to match court model
        ball3d[:,1] = -ball3d[:,1] + 0.5
        
        return ball3d, ball2d
        
    def compute_camera_position(self, fx, fy, dist_coeffs=None, cx=None, cy=None):
        """
        Compute camera position using solvePnP.
        
        Args:
            fx (float): X-axis focal length
            fy (float): Y-axis focal length
            dist_coeffs (ndarray, optional): Distortion coefficients
            cx (float, optional): X center of image (defaults to image center)
            cy (float, optional): Y center of image (defaults to image center)
            
        Returns:
            tuple: (camera_position, success_flag)
        """
        cx = cx or self.image_center[0]
        cy = cy or self.image_center[1]
        
        camera_matrix = np.array([
            [fx, 0,  cx],
            [0,  fy, cy],
            [0,  0,  1]
        ], dtype=np.float32)
        
        if dist_coeffs is None:
            dist_coeffs = np.zeros((5, 1), dtype=np.float32)
            
        obj_pts_pnp = self.object_points.reshape((-1, 1, 3))
        img_pts_pnp = self.image_points.reshape((-1, 1, 2))
        
        # Solve for camera pose
        success, rvec, tvec = cv2.solvePnP(
            obj_pts_pnp, 
            img_pts_pnp, 
            camera_matrix, 
            dist_coeffs, 
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return None, False
        
        # Convert rotation vector to matrix and calculate camera position
        R, _ = cv2.Rodrigues(rvec)
        camera_position = -R.T @ tvec
        
        return camera_position.flatten(), success
        
    def cost_function(self, fx, fy, dist_coeffs=None, w_vertical=1.0, w_dist=1.0, w_proj=0.1, use_reproj=False):
        """
        Calculate optimization cost based on:
        1. Vertical position error
        2. Distance measurement errors
        3. Optional reprojection error
        
        Args:
            fx (float): X-axis focal length
            fy (float): Y-axis focal length
            dist_coeffs (ndarray, optional): Distortion coefficients
            w_vertical (float): Weight for vertical error
            w_dist (float): Weight for distance errors
            w_proj (float): Weight for reprojection error
            use_reproj (bool): Include reprojection error in cost
            
        Returns:
            float: Total cost value
        """
        cx, cy = self.image_center
        
        # Calculate camera position
        cam_pos, success = self.compute_camera_position(fx, fy, dist_coeffs, cx, cy)
        if not success or cam_pos is None:
            return 1e10  # High cost for failed calculations
        
        # Calculate vertical error
        vertical_error = cam_pos[1] - self.vertical_distance
        
        # Calculate distance errors
        dist_errors = []
        for i in range(len(self.measure_points_3d)):
            d_est = np.linalg.norm(cam_pos - self.measure_points_3d[i])
            d_err = d_est - self.measure_distances[i]
            dist_errors.append(d_err)
            
        # Calculate reprojection error if requested
        proj_error = np.array([0.0])
        if use_reproj:
            camera_matrix = np.array([
                [fx, 0,  cx],
                [0,  fy, cy],
                [0,  0,  1]
            ], dtype=np.float32)
            
            success, rvec, tvec = cv2.solvePnP(self.object_points, self.image_points, camera_matrix, dist_coeffs)
            if success:
                R_cam, _ = cv2.Rodrigues(rvec)
                extrinsic_matrix = np.hstack((R_cam, tvec))
                projected_points, _ = cv2.projectPoints(
                    self.object_points, 
                    extrinsic_matrix[:, :3], 
                    extrinsic_matrix[:, 3], 
                    camera_matrix, 
                    dist_coeffs
                )
                projected_points = projected_points.reshape(-1, 2)
                proj_error = np.linalg.norm(projected_points - self.image_points, axis=1)
        
        # Compute total cost
        cost_val = (
            w_vertical * (vertical_error**2) + 
            w_dist * sum(e**2 for e in dist_errors) +
            (w_proj * sum(proj_error**2) if use_reproj else 0)
        )
        
        return cost_val
    
    def grid_search_focal_lengths(self, fx_range=(500, 3600), fy_range=(500, 3100), steps=20):
        """
        Perform grid search to find optimal focal lengths.
        
        Args:
            fx_range (tuple): Range for fx focal length search (min, max)
            fy_range (tuple): Range for fy focal length search (min, max)
            steps (int): Number of steps in each dimension
            
        Returns:
            tuple: (best_fx, best_fy, min_cost)
        """
        best_fx, best_fy = None, None
        min_cost = float('inf')
        
        fx_min, fx_max = fx_range
        fy_min, fy_max = fy_range
        
        print(f'Starting grid search for focal lengths ({steps}x{steps} grid)...')
        for i in tqdm(range(steps), desc='Grid search'):
            fx = fx_min + (fx_max - fx_min) * i / (steps - 1)
            for j in range(steps):
                fy = fy_min + (fy_max - fy_min) * j / (steps - 1)
                cost_val = self.cost_function(fx, fy)
                
                if cost_val < min_cost:
                    min_cost = cost_val
                    best_fx, best_fy = fx, fy
        
        # Verify result with camera position calculation
        cam_pos, success = self.compute_camera_position(best_fx, best_fy)
        if success:
            print(f'Best focal lengths: fx={best_fx:.2f}, fy={best_fy:.2f} (cost={min_cost:.4f})')
            print(f'Resulting camera position: {cam_pos}')
            
            # Calculate errors
            vertical_error = cam_pos[1] - self.vertical_distance
            dist_errors = []
            dist_vals = []
            for i in range(len(self.measure_points_3d)):
                d_est = np.linalg.norm(cam_pos - self.measure_points_3d[i])
                d_err = d_est - self.measure_distances[i]
                dist_errors.append(d_err)
                dist_vals.append(d_est)
                
            print(f'Distance measurements: {dist_vals}')
            print(f'Distance errors: {dist_errors}')
            print(f'Vertical error: {vertical_error}')
            
        return best_fx, best_fy, min_cost
    
    def evaluate_distortion_cost(self, params, fx, fy):
        """
        Evaluate cost function for distortion parameters.
        
        Args:
            params (list): [k1, k2, p1, p2] distortion parameters
            fx (float): X-axis focal length
            fy (float): Y-axis focal length
            
        Returns:
            float: Cost value
        """
        k1, k2, p1, p2 = params
        dist_coeffs = np.array([k1, k2, p1, p2, 0], dtype=np.float32).reshape(5, 1)
        return self.cost_function(fx, fy, dist_coeffs=dist_coeffs, use_reproj=True)
    
    def grid_search_distortion(self, fx, fy, ranges, steps=10, n_jobs=4):
        """
        Parallel grid search for optimal distortion coefficients.
        
        Args:
            fx (float): Fixed X-axis focal length
            fy (float): Fixed Y-axis focal length
            ranges (list): List of (min, max) ranges for [k1, k2, p1, p2]
            steps (int): Number of steps for each parameter
            n_jobs (int): Number of parallel jobs
            
        Returns:
            tuple: (best_params, min_cost)
        """
        k1_range, k2_range, p1_range, p2_range = ranges
        k1_vals = np.linspace(k1_range[0], k1_range[1], steps)
        k2_vals = np.linspace(k2_range[0], k2_range[1], steps)
        p1_vals = np.linspace(p1_range[0], p1_range[1], steps)
        p2_vals = np.linspace(p2_range[0], p2_range[1], steps)
        
        print(f"Starting parallel grid search for distortion coefficients...")
        total_evaluations = steps ** 4
        print(f"Total parameter combinations to evaluate: {total_evaluations}")
        
        # Create a partial function with fixed parameters
        def evaluate_params(k1, k2, p1, p2):
            return self.evaluate_distortion_cost([k1, k2, p1, p2], fx, fy)
        
        # Run parallel grid search
        costs = Parallel(n_jobs=n_jobs)(
            delayed(evaluate_params)(k1, k2, p1, p2)
            for k1, k2, p1, p2 in tqdm(
                product(k1_vals, k2_vals, p1_vals, p2_vals),
                total=total_evaluations,
                desc="Evaluating distortion parameters"
            )
        )
        
        # Find best parameters
        min_cost = float('inf')
        best_params = None
        
        for cost, (k1, k2, p1, p2) in zip(costs, product(k1_vals, k2_vals, p1_vals, p2_vals)):
            if cost < min_cost:
                min_cost = cost
                best_params = [k1, k2, p1, p2]
        
        print(f"Best distortion parameters: {best_params} (cost={min_cost:.4f})")
        return best_params, min_cost
    
    def refine_parameters(self, fx, fy, dist_init=None):
        """
        Refine parameters using Nelder-Mead optimization.
        
        Args:
            fx (float): Initial X-axis focal length
            fy (float): Initial Y-axis focal length
            dist_init (list, optional): Initial distortion coefficients
            
        Returns:
            tuple: (optimized_fx, optimized_fy, optimized_dist)
        """
        if dist_init is None:
            dist_init = [0.0, 0.0, 0.0, 0.0]  # k1, k2, p1, p2
            
        # Create an objective function for focal lengths and distortion
        def full_objective(params):
            fx, fy, k1, k2, p1, p2 = params
            dist_coeffs = np.array([k1, k2, p1, p2, 0], dtype=np.float32).reshape(5, 1)
            return self.cost_function(fx, fy, dist_coeffs=dist_coeffs, use_reproj=True)
            
        initial_params = [fx, fy] + dist_init
        print("Refining parameters with Nelder-Mead optimization...")
        
        result = minimize(
            full_objective,
            initial_params,
            method='Nelder-Mead',
            options={'maxiter': 1000}
        )
        
        if result.success:
            opt_fx, opt_fy, opt_k1, opt_k2, opt_p1, opt_p2 = result.x
            opt_dist = [opt_k1, opt_k2, opt_p1, opt_p2]
            print(f"Refined parameters: fx={opt_fx:.2f}, fy={opt_fy:.2f}")
            print(f"Distortion: {opt_dist}")
            return opt_fx, opt_fy, opt_dist
        else:
            print("Optimization failed to converge. Using initial parameters.")
            return fx, fy, dist_init
    
    def calibrate(self):
        """
        Run the full calibration pipeline.
        
        Returns:
            bool: Success flag
        """
        # Stage 1: Find optimal focal lengths
        best_fx, best_fy, _ = self.grid_search_focal_lengths(
            fx_range=(500, 3600),
            fy_range=(500, 3100),
            steps=30
        )
        
        # Stage 2: Find optimal distortion coefficients
        initial_ranges = [
            (-0.5, 0.5),   # k1 range
            (-0.5, 0.5),   # k2 range
            (-0.1, 0.1),   # p1 range
            (-0.1, 0.1)    # p2 range
        ]
        
        best_dist, _ = self.grid_search_distortion(
            best_fx,
            best_fy,
            ranges=initial_ranges,
            steps=15,
            n_jobs=8  # Adjust based on your CPU
        )
        
        # Optional: Fine-tune all parameters
        # refined_fx, refined_fy, refined_dist = self.refine_parameters(best_fx, best_fy, best_dist)
        
        # Set final calibration parameters
        cx, cy = self.image_center
        self.camera_matrix = np.array([
            [best_fx, 0,      cx],
            [0,      best_fy, cy],
            [0,      0,       1]
        ], dtype=np.float32)
        
        self.dist_coeffs = np.array([best_dist[0], best_dist[1], best_dist[2], best_dist[3], 0], 
                                   dtype=np.float32).reshape(5, 1)
        
        # Compute extrinsic parameters
        success, rvec, tvec = cv2.solvePnP(
            self.object_points, 
            self.image_points, 
            self.camera_matrix, 
            self.dist_coeffs
        )
        
        if success:
            R_cam, _ = cv2.Rodrigues(rvec)
            self.extrinsic_matrix = np.hstack((R_cam, tvec))
            camera_position = (-R_cam.T @ tvec).flatten()
            print("Final camera position:", camera_position)
            return True
        else:
            print("Failed to compute extrinsic parameters with final calibration")
            return False
            
    def save_results(self, extrinsic_path, intrinsic_path):
        """
        Save calibration results to files.
        
        Args:
            extrinsic_path (str): Path to save extrinsic matrix
            intrinsic_path (str): Path to save intrinsic parameters
            
        Returns:
            bool: Success flag
        """
        if self.camera_matrix is None or self.dist_coeffs is None or self.extrinsic_matrix is None:
            print("Cannot save results: Calibration not completed")
            return False
            
        # Save extrinsic matrix
        np.save(extrinsic_path, self.extrinsic_matrix)
        print(f"Extrinsic matrix saved to {extrinsic_path}")
        
        # Extract and save intrinsic parameters as 1x9 array
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        k1, k2, p1, p2, k3 = self.dist_coeffs.ravel()
        
        intrinsics = np.array([fx, fy, cx, cy, k1, k2, k3, p1, p2])
        np.save(intrinsic_path, intrinsics)
        print(f"Intrinsic parameters saved to {intrinsic_path}")
        
        return True
        
    def visualize_results(self, image_path, output_dir="./"):
        """
        Generate visualizations of calibration results.
        
        Args:
            image_path (str): Path to background image for 2D visualization
            output_dir (str): Directory to save visualization files
            
        Returns:
            bool: Success flag
        """
        if self.camera_matrix is None or self.dist_coeffs is None or self.extrinsic_matrix is None:
            print("Cannot visualize: Calibration not completed")
            return False
            
        # 3D visualization of camera pose
        self.visualize_camera_pose(f"{output_dir}/camera_pose.html")
        
        # 2D projection visualization
        self.visualize_2d_projection(image_path, f"{output_dir}/reprojection.png")
        
        return True
        
    def visualize_camera_pose(self, output_file):
        """
        Create 3D visualization of camera pose and court model.
        
        Args:
            output_file (str): Path to save HTML visualization
            
        Returns:
            bool: Success flag
        """
        # Calculate camera position from extrinsic matrix
        R_cam = self.extrinsic_matrix[:, :3]
        tvec = self.extrinsic_matrix[:, 3].reshape(3, 1)
        camera_position = (-R_cam.T @ tvec).flatten()
        
        # Create 3D visualization
        fig = go.Figure()
        
        # Plot 3D world points (court model)
        for i, point in enumerate(self.object_points):
            fig.add_trace(go.Scatter3d(
                x=[point[0]], y=[point[1]], z=[point[2]],
                mode='markers+text',
                text=[f"Point {i+1}"],
                textposition="top center",
                marker=dict(size=4, color='white'),
                name=f"World Point {i+1}"
            ))
        
        # Plot camera position
        fig.add_trace(go.Scatter3d(
            x=[camera_position[0]], y=[camera_position[1]], z=[camera_position[2]],
            mode='markers+text',
            text=["Camera"],
            textposition="bottom center",
            marker=dict(size=6, color='blue'),
            name="Camera Position"
        ))
        
        # Visualize camera orientation with coordinate axes
        axis_length = 1
        origin = camera_position
        R_T = R_cam.T
        
        for i, (axis, color, name) in enumerate(zip(
            [R_T[:, 0], R_T[:, 1], R_T[:, 2]], 
            ['red', 'green', 'blue'], 
            ['X-axis', 'Y-axis', 'Z-axis']
        )):
            end_point = origin + axis * axis_length
            fig.add_trace(go.Scatter3d(
                x=[origin[0], end_point[0]],
                y=[origin[1], end_point[1]],
                z=[origin[2], end_point[2]],
                mode='lines',
                line=dict(color=color, width=4),
                name=name
            ))
        
        # Add court plane
        plane_x = np.array([[3.05, 3.05], [-3.05, -3.05]])
        plane_y = np.array([[0.5, 0.5], [0.5, 0.5]])
        plane_z = np.array([[-6.7, 6.7], [-6.7, 6.7]])
        
        fig.add_trace(go.Surface(
            x=plane_x,
            y=plane_y,
            z=plane_z,
            colorscale=[[0, 'green'], [1, 'green']],
            opacity=0.5,
            showscale=False
        ))
        
        # Configure layout
        fig.update_layout(
            scene=dict(
                xaxis=dict(title='X (meters)'),
                yaxis=dict(title='Y (meters)'),
                zaxis=dict(title='Z (meters)'),
                aspectmode='data'
            ),
            title="Camera Position and Orientation",
        )
        
        # Save visualization
        pio.write_html(fig, file=output_file)
        print(f"3D visualization saved to {output_file}")
        return True
        
    def visualize_2d_projection(self, image_path, output_file):
        """
        Create visualization of 2D reprojection results.
        
        Args:
            image_path (str): Path to background image
            output_file (str): Path to save visualization
            
        Returns:
            bool: Success flag
        """
        try:
            # Project 3D points to 2D
            projected_points, _ = cv2.projectPoints(
                self.object_points, 
                self.extrinsic_matrix[:, :3], 
                self.extrinsic_matrix[:, 3], 
                self.camera_matrix, 
                self.dist_coeffs
            )
            projected_points = projected_points.reshape(-1, 2)
            
            # Load background image
            img = np.array(Image.open(image_path))
            img_height, img_width, _ = img.shape
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(img_width/100, img_height/100), dpi=100)
            ax.imshow(img, extent=[0, img_width, img_height, 0])
            
            # Plot original image points
            ax.scatter(
                self.image_points[:, 0], 
                self.image_points[:, 1], 
                c='blue', 
                label='Original Points', 
                marker='o', 
                s=40
            )
            
            # Add point labels
            for i, (x, y) in enumerate(self.image_points):
                ax.text(x, y, f"{i}", fontsize=8, color='blue')
            
            # Plot projected points
            ax.scatter(
                projected_points[:, 0], 
                projected_points[:, 1], 
                c='red', 
                label='Projected Points', 
                marker='x', 
                s=40
            )
            
            # Draw lines between corresponding points
            for orig, proj in zip(self.image_points, projected_points):
                ax.plot(
                    [orig[0], proj[0]], 
                    [orig[1], proj[1]], 
                    'gray', 
                    linestyle='--', 
                    linewidth=1
                )
            
            # Add labels and legend
            ax.set_title("Reprojection Results")
            ax.set_xlabel("X (pixels)")
            ax.set_ylabel("Y (pixels)")
            ax.legend()
            
            # Adjust axis limits
            ax.set_xlim([0, img_width])
            ax.set_ylim([img_height, 0])
            
            # Save visualization
            plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
            print(f"2D visualization saved to {output_file}")
            return True
            
        except Exception as e:
            print(f"Error creating 2D visualization: {e}")
            return False


def parse_arguments():
    """Parse command-line arguments for camera calibration."""
    parser = argparse.ArgumentParser(description="Camera Calibration Tool")
    
    parser.add_argument(
        '--court_points',
        type=str,
        required=True,
        help="Path to the pickle file containing court coordinates"
    )
    
    parser.add_argument(
        '--measurements',
        type=str,
        required=True,
        help="Path to the JSON file containing distance measurements"
    )
    
    parser.add_argument(
        '--camera_id',
        type=str,
        required=True,
        help="Camera ID (e.g. '0', '2', '7')"
    )
    
    parser.add_argument(
        '--image',
        type=str,
        default=None,
        help="Path to reference image for visualization"
    )
    
    parser.add_argument(
        '--ball_3d',
        type=str,
        default=None,
        help="Path to 3D ball positions CSV (optional)"
    )
    
    parser.add_argument(
        '--ball_2d',
        type=str,
        default=None,
        help="Path to 2D ball detections CSV (optional)"
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default="./",
        help="Directory to save calibration results and visualizations"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for camera calibration application."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Configure output paths
    output_dir = args.output_dir
    extrinsic_path = f"{output_dir}/Cam_{args.camera_id}_extrinsic.npy"
    intrinsic_path = f"{output_dir}/Cam_{args.camera_id}_intrinsic.npy"
    
    # Create calibrator instance
    calibrator = CameraCalibrator()
    
    # # Load and check the structure of your pickle file
    # with open('./source_data/view4/4.pkl', 'rb') as f:
    #     data = pickle.load(f)
    #     print(type(data))
    #     print(data)  # Print the content to see its structure
        
    # Load data
    print(f"Loading data for camera {args.camera_id}...")
    if not calibrator.load_data(
        args.court_points,
        args.measurements,
        args.camera_id,
        args.ball_3d,
        args.ball_2d
    ):
        print("Failed to load data. Exiting.")
        return 1
    
    # Run calibration
    print("Starting calibration process...")
    if not calibrator.calibrate():
        print("Calibration failed. Exiting.")
        return 1
    
    # Save results
    calibrator.save_results(extrinsic_path, intrinsic_path)
    
    # Generate visualizations if image is provided
    if args.image:
        calibrator.visualize_results(args.image, output_dir)
    
    print("Calibration completed successfully.")
    return 0


if __name__ == "__main__":
    main()

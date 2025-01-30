import numpy as np
from scipy.spatial.transform import Rotation as R

def xyaxes_to_quaternion(xyaxes):
    """
    Converts an xy-axes representation (six numbers) to a quaternion.
    Assumes the input format is [x1, x2, x3, y1, y2, y3].
    
    Parameters:
        xyaxes (list or array): A list or numpy array containing six elements.
    
    Returns:
        np.array: A quaternion [x, y, z, w] representing the rotation.
    """
    # Extract x-axis and y-axis vectors
    x_axis = np.array(xyaxes[:3])
    y_axis = np.array(xyaxes[3:])
    
    # Compute z-axis using the cross product
    z_axis = np.cross(x_axis, y_axis)
    
    # Normalize the basis vectors to ensure an orthonormal frame
    x_axis /= np.linalg.norm(x_axis)
    y_axis /= np.linalg.norm(y_axis)
    z_axis /= np.linalg.norm(z_axis)
    
    # Construct the rotation matrix
    R_matrix = np.column_stack((x_axis, y_axis, z_axis))
    
    # Convert rotation matrix to quaternion
    quaternion = R.from_matrix(R_matrix).as_quat()  # [x, y, z, w] format
    
    return quaternion

if __name__ == "__main__":
    xyaxes = [1.000, -0.024, -0.000, 0.018, 0.775, 0.631]
    quaternion = xyaxes_to_quaternion(xyaxes)
    print("Quaternion:", quaternion)

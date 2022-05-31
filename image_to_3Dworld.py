from math import cos,sin,tan,pi
import numpy as np
from shapely.geometry import Point
from shapely.affinity import translate


def rotate_x(angle):
    R = np.mat([[ 1,         0,           0],
                 [ 0, cos(angle), -sin(angle) ],
                 [ 0, sin(angle),  cos(angle) ]])
    return R


def rotate_y(angle):
    R = np.mat([[ cos(angle), 0,  sin(angle) ],
                 [ 0,         1,          0 ],
                 [-sin(angle), 0,  cos(angle) ]])
    return R


def rotate_z(angle):
    R = np.mat([[ cos(angle), -sin(angle), 0 ],
                 [ sin(angle),  cos(angle), 0 ],
                 [         0,          0, 1 ]])
    return R


def to_rot(r):
    return rotate_x(r[0]) * rotate_y(r[1]) * rotate_y(r[2])


class Image2World():
    """Facility class to convert image point to 3d World point.
    """

    def __init__(self, camera_params):
        """Initialize a converter from image point to 3d point.

        Args:
            camera_params(dict): A dictionary with the camera information.
        """
        self._cam_x = camera_params['x']
        self._cam_y = camera_params['y']
        self._cam_z = camera_params['z'] 
        self._cam_pitch = camera_params['pitch']
        self._cam_roll = camera_params['roll']
        self._cam_yaw = camera_params['yaw']
        self._cam_width = camera_params['width'] 
        self._cam_height = camera_params['height'] 
        self._cam_fov = camera_params['fov']

        self._set_intrinsic_matrix()

    def _image_to_camera_frame(self, object_camera_frame, car_yaw):
        """Transform the pixel from image frame to camera frame.

        Args:
            object_camera_frame (np.ndarray): A 4x1 array with [x;y;z;1] where z is probably 1.

        Return: The point projected into the camera frame.
        """
        rotation_image_camera_frame = np.dot(rotate_z(car_yaw + 90 * pi / 180), rotate_x(90 * pi / 180))
        image_camera_frame = np.zeros((4,4))
        image_camera_frame[:3,:3] = rotation_image_camera_frame
        image_camera_frame[:, -1] = [0, 0, 0, 1]

        return np.dot(image_camera_frame, object_camera_frame)

    def _set_intrinsic_matrix(self):
        """Compute the intrinsic matrix starting from the camera parameters.
        """
        f = self._cam_width / (2 * tan(self._cam_fov * pi / 360))
        Center_X = self._cam_width / 2.0
        Center_Y = self._cam_height / 2.0

        intrinsic_matrix = np.array([[f, 0, Center_X],
                                     [0, f, Center_Y],
                                     [0, 0, 1       ]])

        self._inv_intrinsic_matrix = np.linalg.inv(intrinsic_matrix)

    def convert(self, pixel_xy, depth, car_x, car_y, car_z, car_yaw):
        """
        Get a point in the image and return the 3D world coordinates.

        Args:
            pixel_xy: (x,y) point in the image.
            depth: depth of the pixel_xy point in meters.
            car_x: position of the car on x axes in meters
            car_y: position of the car on y axes in meters
            car_z: position of the car on z axes in meters
            car_yaw: rotation of teh car respect to the xy plane.

        Return:
            position: [x,y,z] representing that point in the 3D world.
        """
        pixel_xy = np.reshape(np.array([*pixel_xy, 1]), (3,1))

        # Projection Pixel to Image Frame
        image_frame_vect = np.dot(self._inv_intrinsic_matrix, pixel_xy) * depth
        
        # Create extended vector
        image_frame_vect_extended = np.zeros((4,1))
        image_frame_vect_extended[:3] = image_frame_vect 
        image_frame_vect_extended[-1] = 1

        # Projection Camera to Vehicle Frame
        camera_frame = self._image_to_camera_frame(image_frame_vect_extended, car_yaw)
        camera_frame = camera_frame[:3]
        camera_frame = np.asarray(np.reshape(camera_frame, (1,3)))

        camera_frame_extended = np.zeros((4,1))
        camera_frame_extended[:3] = camera_frame.T 
        camera_frame_extended[-1] = 1

        camera_to_vehicle_frame = np.zeros((4,4))
        # Take into account the rotation of the camera respect to the car and the rotation of the car.
        # This works because the car does not rotate on pitch and roll, so yaw is coherent.
        camera_to_vehicle_frame[:3,:3] = to_rot([self._cam_roll, self._cam_pitch, self._cam_yaw])
        # Take into account the rotation of the car and the position.
        rot_x = cos(car_yaw) * (self._cam_x - 0) - sin(car_yaw) * (self._cam_y - 0) + 0
        rot_y = sin(car_yaw) * (self._cam_x - 0) + cos(car_yaw) * (self._cam_y - 0) + 0
        camera_to_vehicle_frame[:,-1] = [car_x + rot_x, car_y + rot_y, car_z + self._cam_z, 1]

        vehicle_frame = np.dot(camera_to_vehicle_frame, camera_frame_extended)
        vehicle_frame = vehicle_frame[:3]
        vehicle_frame = np.asarray(np.reshape(vehicle_frame, (1,3)))

        return vehicle_frame[0]


# DRIVER TEST            
if __name__ == "__main__":
    camera_parameters = {}
    camera_parameters['x'] = 0 
    camera_parameters['y'] = 0.0
    camera_parameters['z'] = 0 
    camera_parameters['pitch'] = 0.0
    camera_parameters['roll'] = 0.0
    camera_parameters['yaw'] = 0.0
    camera_parameters['width'] = 200 
    camera_parameters['height'] = 200 
    camera_parameters['fov'] = 90

    ego_x = 100
    ego_y = 200
    ego_z = 0
    ego_yaw = -pi/4

    depth = 5
    pixel_xy = [10,100]
    c = Image2World(camera_parameters)
    print([round(x,2) for x in c.convert(pixel_xy, depth, ego_x, ego_y, ego_z, ego_yaw)])

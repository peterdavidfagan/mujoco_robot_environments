"""Scene visualization utilities."""
from typing import Tuple
import numpy as np

from mujoco_robot_environments.environment.props import Prop
from dm_control import composer, mjcf, mujoco
import PIL.Image


# this camera class is take from dm_robotics: https://github.com/google-deepmind/dm_robotics/blob/main/py/moma/prop.py
# the rest of this file is custom code
class Camera(Prop):
  """Base class for Moma camera props."""

  def _build(  # pylint:disable=arguments-renamed  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
      self,
      name: str,
      mjcf_root: mjcf.RootElement,
      camera_element: str,
      prop_root: str = 'prop_root',
      width: int = 480,
      height: int = 640,
      fovy: float = 90.0):
    """Camera  constructor.

    Args:
      name: The unique name of this prop.
      mjcf_root: The root element of the MJCF model.
      camera_element: Name of the camera MJCF element.
      prop_root: Name of the prop root body MJCF element.
      width: Width of the camera image.
      height: Height of the camera image.
      fovy: Field of view, in degrees.
    """
    super()._build(name, mjcf_root, prop_root)

    self._camera_element = camera_element
    self._width = width
    self._height = height
    self._fovy = fovy

    # Sub-classes should extend `_build` to construct the appropriate mjcf, and
    # over-ride the `rgb_camera` and `depth_camera` properties.

  @property
  def camera(self) -> mjcf.Element:
    """Returns an mjcf.Element representing the camera."""
    return self._mjcf_root.find('camera', self._camera_element)

  def get_camera_pos(self, physics: mjcf.Physics) -> np.ndarray:
    return physics.bind(self.camera).xpos  # pytype: disable=attribute-error

  def get_camera_quat(self, physics: mjcf.Physics) -> np.ndarray:
    return tr.mat_to_quat(
        np.reshape(physics.bind(self.camera).xmat, [3, 3]))  # pytype: disable=attribute-error

  def render_rgb(self, physics: mjcf.Physics) -> np.ndarray:
    return np.atleast_3d(
        physics.render(
            height=self._height,
            width=self._width,
            camera_id=self.camera.full_identifier,  # pytype: disable=attribute-error
            depth=False))

  def render_depth(self, physics: mjcf.Physics) -> np.ndarray:
    return np.atleast_3d(physics.render(
        height=self._height,
        width=self._width,
        camera_id=self.camera.full_identifier,  # pytype: disable=attribute-error
        depth=True))

  def get_intrinsics(self, physics: mjcf.Physics) -> np.ndarray:
    focal_len = self._height / 2 / np.tan(self._fovy / 2 * np.pi / 180)
    return np.array([[focal_len, 0, (self._height - 1) / 2, 0],
                     [0, focal_len, (self._height - 1) / 2, 0],
                     [0, 0, 1, 0]])


def _make_fixed_camera(
    name: str,
    pos: Tuple = (0.0, 0.0, 0.0),
    quat: Tuple = (0.0, 0.0, 0.0, 1.0),
    height: int = 640,
    width: int = 480,
    fovy: float = 90.0,
) -> None:
    """Create fixed camera."""
    mjcf_root = mjcf.element.RootElement(model=name)
    prop_root = mjcf_root.worldbody.add(
        "body",
        name=f"{name}_root",
    )
    camera = prop_root.add(
        "camera",
        name=name,
        mode="fixed",
        pos=pos,
        quat=quat,
        fovy=fovy,
    )

    return mjcf_root, camera


class FixedCamera(Camera):
    """Fixed camera."""

    def _build(
        self,
        name: str,
        pos: str = "0 0 0",
        quat: str = "0 0 0 1",
        height: int = 640,
        width: int = 480,
        fovy: float = 90.0,
    ) -> None:
        """Build the camera."""
        # make the mjcf element
        mjcf_root, camera = _make_fixed_camera(
            name,
            pos,
            quat,
            height,
            width,
            fovy,
        )

        # build the camera
        super()._build(
            name=name,
            mjcf_root=mjcf_root,
            camera_element=name,
            prop_root=f"{name}_root",
            width=width,
            height=height,
            fovy=fovy,
        )
        del camera


def add_camera(
    arena: composer.Arena,
    name: str,
    pos: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    quat: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    height: int = 480,
    width: int = 480,
    fovy: float = 90.0,
) -> composer.Entity:
    """Add a camera to the arena."""
    # create fixed camera
    camera = FixedCamera(
        name=name,
        pos=pos,
        quat=quat,
        height=height,
        width=width,
        fovy=fovy,
    )

    # attach to arena
    arena.mjcf_model.attach(camera.mjcf_model)

    # TODO: investigate, strangely find_all and find result in different results
    cameras = camera_prop = arena.mjcf_model.find_all("camera")
    for camera in cameras:
        if camera.name == name:
            camera_prop = camera
            break

    return camera


def render_scene(physics: mjcf.Physics, x=0.0, y=0.0, z=0.0, roll=2.5, pitch=180, yaw=-30) -> None:
    """Render the scene using a movable camera."""
    camera = mujoco.MovableCamera(physics, height=480, width=480)
    camera.set_pose([x, y, z], roll, pitch, yaw)
    image_arr = camera.render()
    image = PIL.Image.fromarray(image_arr)
    image.show()

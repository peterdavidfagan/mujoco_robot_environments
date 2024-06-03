"""A script for defining props."""
import os
from dataclasses import dataclass, field
from pathlib import Path
import random
from typing import Dict, Sequence, Tuple, List

from dm_control import composer, mjcf, utils

import numpy as np


COLOURS= {
    "red": (1.0, 0.0, 0.0, 1.0),
    "green": (0.0, 1.0, 0.0, 1.0),
    "blue": (0.0, 0.0, 1.0, 1.0),
    "yellow": (1.0, 1.0, 0.0, 1.0),
    "cyan": (0.0, 1.0, 1.0, 1.0),
    "magenta": (1.0, 0.0, 1.0, 1.0),
}

filepath = Path(__file__).resolve().parent.absolute()
TEXTURES = {}
TEXTURE_PATH = f"{filepath}/../assets/textures"
for name in os.listdir(TEXTURE_PATH):
    textures = []
    textures_path = os.path.join(TEXTURE_PATH,name)
    for texture in os.listdir(textures_path):
        texture_path = os.path.join(textures_path, texture)
        textures.append(mjcf.Asset(utils.io.GetResource(texture_path), ".png"))
    TEXTURES[name] = textures

MJCFS = {}
MJCF_PATH = f"{filepath}/../assets/mjcf"
for name in os.listdir(MJCF_PATH):
    mjcf_path = os.path.join(MJCF_PATH, name)
    for model in os.listdir(mjcf_path):
        if model.endswith('.xml'):
            model_path = os.path.join(mjcf_path, model)

    MJCFS[name] = model_path

@dataclass
class PropsLabels:
    """Container for prop labels."""
    data: dict = field(default_factory=dict)
    default_values: dict = field(default_factory=lambda: {"texture": "plain"})
    
    def __post_init__(self):
        # Set default values
        for key, value in self.default_values.items():
            setattr(self, key, value)
        
        # Override with values from the input dictionary
        for key, value in self.data.items():
            setattr(self, key, value)

    def __str__(self):
        attrs = ', '.join(f"{key}='{value}'" for key, value in self.__dict__.items() if key not in {'data', 'default_values'})
        return f"PropsLabels({attrs})"

    def __repr__(self):
        return self.__str__()
    
# this prop class is take from dm_robotics: https://github.com/google-deepmind/dm_robotics/blob/main/py/moma/prop.py
# the rest of this file is custom code
class Prop(composer.Entity):

    """Base class for MOMA props."""
    def _build(self,
                name: str,
                mjcf_root: mjcf.RootElement,
                prop_root: str = 'prop_root',
                labels: PropsLabels = PropsLabels()) -> None:
        """Base constructor for props.

        This constructor sets up the common observables and element access
        properties.

        Args:
            name: The unique name of this prop.
            mjcf_root: (mjcf.Element) The root element of the MJCF model.
            prop_root: (string) Name of the prop root body MJCF element.

        Raises:
            ValueError: If the model does not contain the necessary elements.
        """

        self._name = name
        self._mjcf_root = mjcf_root
        self._prop_root = mjcf_root.find('body', prop_root)  # type: mjcf.Element
        if self._prop_root is None:
            raise ValueError(f'model does not contain prop root {prop_root}.')
        self._freejoint = None  # type: mjcf.Element
        self._labels = labels

    @property
    def name(self) -> str:
        return self._name

    @property
    def mjcf_model(self) -> mjcf.RootElement:
        """Returns the `mjcf.RootElement` object corresponding to this prop."""
        return self._mjcf_root
    
    @property
    def labels(self) -> PropsLabels:
        return self._labels

    def set_pose(self, physics: mjcf.Physics, position: np.ndarray,  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
                quaternion: np.ndarray) -> None:
        """Sets the pose of the prop wrt to where it was defined.

        This function overrides `Entity.set_pose`, which has the annoying property
        that it doesn't consider where the prop was originally attached.  EG if you
        do `pinch_site.attach(prop)`, the prop will be a sibling of pinch_site with
        the pinch_site's pose parameters, and calling
            `set_pose([0, 0, 0], [1, 0, 0, 0])`
        will clobber these params and move the prop to the parent-body origin.

        Oleg's fix uses an extra `prop_root` body that's a child of the sibling
        body, and sets the pose of this guy instead.

        Args:
            physics: An instance of `mjcf.Physics`.
            position: A NumPy array of size 3.
            quaternion: A NumPy array of size [w, i, j, k].

        Raises:
            RuntimeError: If the entity is not attached.
            Exception: If oleg isn't happy
        """

        if self._prop_root is None:
            raise Exception('prop {} missing root element'.format(
                self.mjcf_model.model))

        if self._freejoint is None:
            physics.bind(self._prop_root).pos = position  # pytype: disable=not-writable
            physics.bind(self._prop_root).quat = quaternion  # pytype: disable=not-writable
        else:
            # If we're attached via a freejoint then bind().pos or quat does nothing,
            # as the pose is controlled by qpos directly.
            physics.bind(self._freejoint).qpos = np.hstack([position, quaternion])  # pytype: disable=not-writable

    def set_freejoint(self, joint: mjcf.Element):
        """Associates a freejoint with this prop if attached to arena."""
        joint_type = joint.tag  # pytype: disable=attribute-error
        if joint_type != 'freejoint':
            raise ValueError(f'Expected a freejoint but received {joint_type}')
        self._freejoint = joint

    def disable_collisions(self) -> None:
        for geom in self.mjcf_model.find_all('geom'):
            geom.contype = 0
            geom.conaffinity = 0

    @staticmethod
    def _make_material(texture, mjcf):
        assert texture in TEXTURES
        """add material node"""
        tex = mjcf.asset.add("texture",
                            file=random.choice(TEXTURES[texture]),
                            type="skybox",
                            name="texture")
        return mjcf.asset.add("material",
                            name="material",
                            texture=tex,
                            )

class Rectangle(Prop):
    """Prop with a rectangular shape."""

    @staticmethod
    def _make(
        name:str,
        pos: Tuple[float, float, float]=(0.0, 0.0, 0.0),
        x_len: float = 0.1,
        y_len: float = 0.1,
        z_len: float = 0.1,
        rgba: Tuple[float, float, float,float]=(1, 0, 0, 1),
        texture: str = "plain",
        friction: Tuple[float, float, float]=(1, 0.005, 0.0001),
        solimp: Tuple[float, float, float]=(0.95, 0.995, 0.001, 0.5, 3),
        solref: Tuple[float, float, float]=(0.04, 1.1),
        mass: float = 0.15,
        margin: float = 0.15,
        gap: float = 0.15,
    ):
        """Make a block model: the mjcf element, and site."""
        mjcf_root = mjcf.element.RootElement(model=name)
        prop_root = mjcf_root.worldbody.add("body", name="prop_root")
        # add margin so object is pickable
        box = prop_root.add(
            "geom",
            name=name,
            type="box",
            pos=pos,
            material=Prop._make_material(texture, mjcf_root),
            size=(x_len, y_len, z_len),
            solref=solref,
            solimp=solimp,
            condim=3,
            rgba=rgba,
            mass = mass,
            friction = friction,
            margin = margin,
            gap = gap,
        )
        site = prop_root.add(
            "site",
            name="box_centre",
            type="sphere",
            rgba=(0.1, 0.1, 0.1, 0.8),
            size=(0.005,),
            pos=(0, 0, 0),
            euler=(0, 0, 0),
        )  # Was (np.pi, 0, np.pi / 2)
        del box

        return mjcf_root, site

    
    def _build(  # pylint:disable=arguments-renamed
        self,
        rgba: List,
        name: str = "box",
        texture: str = "plain",
        labels: PropsLabels = PropsLabels(),
        x_len: float = 0.1,
        y_len: float = 0.1,
        z_len: float = 0.1,
        pos=(0.0, 0.0, 0.0),
        friction: Tuple[float, float, float]=(1, 0.005, 0.0001),
        solimp: Tuple[float, float, float]=(0.95, 0.995, 0.001),
        solref: Tuple[float, float, float]=(0.002, 0.7),
        mass: float = 0.1,
        margin: float = 0.15,
        gap: float = 0.15,
    ) -> None:
        mjcf_root, site = Rectangle._make(name,
                                          x_len=x_len,
                                          y_len=y_len,
                                          z_len=z_len,
                                          rgba=rgba,
                                          texture=texture,
                                          pos=pos,
                                          solimp=solimp,
                                          solref=solref,
                                          mass=mass,
                                          margin=margin,
                                          gap=gap,
                                          )
        super()._build(name, mjcf_root, "prop_root", labels)
        del site

    @staticmethod
    def _add(
        arena: composer.Arena,
        name: str = "red_rectangle_1",
        colour: str = "red",
        texture: str = "plain",
        labels: PropsLabels = PropsLabels(),
        min_object_size: float = 0.02,
        max_object_size: float = 0.05,     
        x_len: float = 0.04,
        y_len: float = 0.04,
        z_len: float = 0.04,
        rgba: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0),
        sample_size: bool = False,
        sample_colour: bool = False,
        is_cube: bool = False,
        colour_noise: float = 0.1,
    ) -> composer.Entity:
        """Add a block to the arena."""
        if sample_size:
            # sample block dimensions
            if is_cube:
                size = 3*[np.random.uniform(min_object_size, max_object_size)]
            else:
                size = np.random.uniform(min_object_size, max_object_size, size=3)

            x_len, y_len, z_len = size[0], size[1], size[2]
                
        if sample_colour:
            # sample block colour
            rgba = COLOURS[colour]
            # add noise
            rgba = [ c + np.random.uniform(-colour_noise, colour_noise) for c in rgba]
            rgba[3] = 1.0
            
        # create block and add to arena
        rectangle = Rectangle(name=name,
                              texture=texture,
                              x_len=x_len,
                              y_len=y_len,
                              z_len=z_len,
                              rgba=rgba,
                              labels=labels)
        frame = arena.add_free_entity(rectangle)
        rectangle.set_freejoint(frame.freejoint)

        return rectangle
    

class Cylinder(Prop):
    """A cylinder prop."""

    def _make(rgba: List,
            name: str = "cylinder",
            texture: str = "plain",
            radius: float = 0.025,
            half_height: float = 0.1):
        mjcf_root = mjcf.element.RootElement(model=name)
        prop_root = mjcf_root.worldbody.add("body", name="prop_root")
        cylinder = prop_root.add("geom",
                                 name=name,
                                 type="cylinder",
                                 pos=(0, 0, 0),
                                 material=Prop._make_material(texture, mjcf_root),
                                 size=(radius, half_height),
                                 rgba=rgba,
                                 #mass=50
                                 )

        return mjcf_root, cylinder

    def _build(self,
               rgba: List,
               name: str = "cylinder",
               texture: str = "plain",
               radius: float = 0.025,
               half_height: float = 0.1,
               labels: PropsLabels = PropsLabels()) -> None:
        """Build the prop."""
        mjcf_root, cylinder = Cylinder._make(name=name,
                                             radius=radius,
                                             texture = texture,
                                             half_height=half_height,
                                             rgba=rgba)
        super()._build(name, mjcf_root, "prop_root", labels)
        del cylinder

    
    @staticmethod
    def _add(
        arena: composer.Arena,
        name: str = "1",
        colour: str = "red",
        texture: str = "plain",
        labels: PropsLabels = PropsLabels(),
        min_object_size: float = 0.02,
        max_object_size: float = 0.05,     
        radius: float = 0.025,
        half_height: float = 0.1,
        rgba: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0),
        sample_size: bool = False,
        sample_colour: bool = False,
        colour_noise: float = 0.1,
    ) -> composer.Entity:
        if sample_size:
            # sample block dimensions
            size = np.random.uniform(min_object_size, max_object_size, size=2)

            radius, half_height = size[0], size[1]

        if sample_colour:
            # sample block colour
            rgba = COLOURS[colour]
            # add noise
            rgba = [ c + np.random.uniform(-colour_noise, colour_noise) for c in rgba]
            rgba[3] = 1.0
        
        cylinder = Cylinder(name=name,
                            radius=radius,
                            texture=texture,
                            half_height=half_height,
                            rgba=rgba,
                            labels=labels)

        frame = arena.add_free_entity(cylinder)
        cylinder.set_freejoint(frame.freejoint)
        return cylinder


class Sphere(Prop):
    """A sphere prop."""

    @staticmethod
    def _make(rgba: List,
            name: str = "sphere",
            texture: str = "plain",
            radius: float = 0.025):
        mjcf_root = mjcf.element.RootElement(model=name)
        prop_root = mjcf_root.worldbody.add("body", name="prop_root")
        sphere = prop_root.add("geom",
                               name=name,
                               type="sphere",
                               material=Prop._make_material(texture, mjcf_root),
                               pos=(0, 0, 0),
                               size=(radius,),
                               rgba=rgba,
                               #mass=50,
                               )

        return mjcf_root, sphere

    def _build(self,
               rgba: List,
               name: str = "sphere",
               texture: str = "plain",
               radius: float = 0.5,
               labels: list[str]=[]) -> None:
        """Build the prop."""
        mjcf_root, sphere = Sphere._make(name=name,
                                         texture=texture,
                                         radius=radius,
                                         rgba=rgba)
        super()._build(name, mjcf_root, "prop_root", labels)
        del sphere

    @staticmethod
    def _add(
        arena: composer.Arena,
        name: str = "1",
        colour: str = "red",
        texture: str = "plain",
        labels: PropsLabels = PropsLabels(),
        min_object_size: float = 0.02,
        max_object_size: float = 0.05,          
        radius: float = 0.025,
        rgba: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0),
        sample_size: bool = False,
        sample_colour: bool = False,
        colour_noise: float = 0.1,
    ) -> composer.Entity:
        if sample_size:
            # sample block dimensions
            radius = np.random.uniform(min_object_size, max_object_size)
        if sample_colour:
            # sample block colour
            rgba = COLOURS[colour]
            # add noise
            rgba = [ c + np.random.uniform(-colour_noise, colour_noise) for c in rgba]
            rgba[3] = 1.0

        # create sphere and add to arena
        sphere = Sphere(name=name,
                        texture=texture,
                        radius=radius,
                        rgba=rgba,
                        labels=labels)
        frame = arena.add_free_entity(sphere)
        sphere.set_freejoint(frame.freejoint)
        return sphere


class GalaApple(Prop):
    """Gala apple prop."""

    @staticmethod
    def _make():
        mjcf_root = mjcf.from_path(MJCFS['gala_apple'])
        return mjcf_root, None

    
    def _build(  # pylint:disable=arguments-renamed
        self,
        name,
    ) -> None:
        mjcf_root, _ = self._make()
        super()._build(name, mjcf_root, "gala_apple")

    @staticmethod
    def _add(
        arena: composer.Arena,
        name: str = "gala_apple",
    ) -> composer.Entity:
        # create apple and add to arena
        apple = GalaApple(name=name)
        frame = arena.add_free_entity(apple)
        apple.set_freejoint(frame.freejoint)

        return apple

class Tray(Prop):
    """tray prop."""

    @staticmethod
    def _make():
        mjcf_root = mjcf.from_path(MJCFS['tray'])
        return mjcf_root, None

    
    def _build(  # pylint:disable=arguments-renamed
        self,
        name,
    ) -> None:
        mjcf_root, _ = self._make()
        super()._build(name, mjcf_root, "model")

    @staticmethod
    def _add(
        arena: composer.Arena,
        name: str = "tray",
    ) -> composer.Entity:
        # create apple and add to arena
        tray = Tray(name=name)
        frame = arena.add_free_entity(tray)
        tray.set_freejoint(frame.freejoint)

        return tray

def add_object(area: composer.Arena,
               name: str,
               shape: str,
               colour: str,
               texture:str,
               labels: PropsLabels,
               min_object_size: float = 0.02,
               max_object_size: float = 0.05,
               sample_size: bool = False,
               sample_colour: bool = False,
               colour_noise: float=0.1,) -> composer.Entity:
    """Add an object to the arena based on the shape and colour."""
    if shape == "cube":
        return Rectangle._add(area,
                              name,
                              colour,
                              texture,
                              labels,
                              min_object_size,
                              max_object_size,                             
                              is_cube=True,
                              sample_size=sample_size,
                              sample_colour=sample_colour,
                              colour_noise=colour_noise)
    elif shape == "rectangle":
        return Rectangle._add(area,
                              name,
                              colour,
                              texture,
                              labels,
                              min_object_size,
                              max_object_size,           
                              sample_size=sample_size,
                              sample_colour=sample_colour,
                              colour_noise=colour_noise)
    elif shape == "cylinder":
        return Cylinder._add(area,
                             name,
                             colour,
                             texture,
                             labels,
                             min_object_size,
                             max_object_size,          
                             sample_size=sample_size,
                             sample_colour=sample_colour,
                             colour_noise=colour_noise)
    elif shape == "sphere":
        return Sphere._add(area,
                           name,
                           colour,
                           texture,
                           labels,
                           min_object_size,
                           max_object_size,          
                           sample_size=sample_size,
                           sample_colour=sample_colour,
                           colour_noise=colour_noise)
    
    elif shape == "apple":
        return GalaApple._add(area, 
                            name,)
    else:
        raise ValueError(f"Unknown shape {shape}")

def add_objects(
    arena: composer.Arena,
    shapes: List[str], 
    colours: List[str],
    textures: List[str],
    min_object_size: float,
    max_object_size: float,
    min_objects: int,
    max_objects: int,
    sample_size: bool = True,
    sample_colour: bool = True,
    colour_noise: float = 0.1,
) -> List[composer.Entity]:
    """Add objects to the arena."""

    assert all(colour in COLOURS.keys() for colour in colours), "Unknown colour"
    assert all(texture in TEXTURES.keys() for texture in textures), "Unknown texture"

    extra_sensors = []
    props = []

    if min_objects == max_objects:
        num_objects = min_objects
    else:
        num_objects = np.random.randint(min_objects, max_objects)
    
    for i in range(num_objects):

        shape = random.choice(shapes)
        if i > 1:
            colour = random.choice(colours)
        else:
            colour = colours[i]

        texture = "plain" # for now fix to plain
        #random.choice(textures)
    
        name = f"prop_{i}"
        labels = PropsLabels({
            "shape": shape,
            "colour": colour,
            "texture": texture,
        })
        obj = add_object(arena,
                         name,
                         shape,
                         colour,
                         texture,
                         labels,
                         min_object_size,
                         max_object_size,
                         sample_size,
                         sample_colour,
                         colour_noise)
        
        props.append(obj)

    return props

import math
import random
import uuid

import numpy as np
import pygame as pg


from peepo.pp.peepo import Peepo

LEFT = 'left'
RIGHT = 'right'

VISION = 'vision'
MOTOR = 'pro'
TRANSPARENT = (0, 0, 0, 0)


class AntPeepo(Peepo):
    """
    This organism represents peepo.
    """

    SIZE = (4, 4)
    RADIUS = 37
    SPEED = 2

    def __init__(self, name, network, graphical, pos=(0, 0)):
        super().__init__(name=name, network=network, graphical=graphical, pos=pos, size=AntPeepo.SIZE)

        self.path = []

        self.nest_pos = (300, 200)
        self.rotation = 50

    def observation(self, name):
        if "obs_ch" in name:
            return [0.1, 0.9] if self.obs_is_in_quadrant(name, self.obs_get_current_heading()) else [0.9, 0.1]
        if "obs_dh" in name:
            return [0.1, 0.9] if self.obs_is_in_quadrant(name, self.obs_get_desired_heading()) else [0.9, 0.1]
        if "pro" in name:
            return [0.9, 0.1]
        return [0.5, 0.5]

    def action(self, node, prediction):
        if np.argmax(prediction) > 0:
            direction = self.get_direction(node)
            if direction == LEFT:
                self.action_turn_left()
            elif direction == RIGHT:
                self.action_turn_right()

    def obs_get_current_heading(self):
        return self.rotation

    def obs_get_desired_heading(self):
        delta_x = self.nest_pos[0] - self.rect.x
        delta_y = self.nest_pos[1] - self.rect.y
        angle = math.atan2(delta_y, delta_x)
        angle_degrees = math.degrees(angle)
        if angle_degrees < 0:
            angle_degrees += 360
        return angle_degrees

    @staticmethod
    def obs_is_in_quadrant(name, rotation):
        parts = name.split("_")
        if len(parts) == 3 and parts[0] == "obs" and (parts[1] == "ch" or parts[1] == "dh"):
            try:
                quadrant = int(parts[2])
                if 0 <= quadrant <= 7:
                    # Calculate the range of degrees for the quadrant
                    lower_limit = quadrant * 45
                    upper_limit = (quadrant + 1) * 45

                    return lower_limit <= rotation < upper_limit
                else:
                    raise ValueError("Quadrant number must be between 0 and 7.")
            except ValueError:
                return False
        else:
            return False

    def action_turn_left(self):
        self.rotation -= 10
        if self.rotation < 0:
            self.rotation = 360

    def action_turn_right(self):
        self.rotation += 10
        if self.rotation > 360:
            self.rotation = 0

    @staticmethod
    def get_direction(name):
        for direction in [LEFT, RIGHT]:
            if direction.lower() in name.lower():
                return direction
        raise ValueError('Unexpected node name %s, could not find LEFT, RIGHT', name)

    def update(self):
        self.generative_model.process()

        self.rect.x += AntPeepo.SPEED * math.cos(math.radians(self.rotation))
        self.rect.y += AntPeepo.SPEED * math.sin(math.radians(self.rotation))
        self.clip()

        if self.graphical:
            self.image = pg.transform.rotate(self.image_original, -self.rotation)
            self.rect = self.image.get_rect(center=self.rect.center)

        self.path.append((self.rect.x, self.rect.y))

    def draw(self, surface):
        surface.blit(self.image, self.rect)
        pg.draw.circle(surface, (0, 0, 255), (self.nest_pos[0], self.nest_pos[1]), 5)

        for step in self.path:
            surface.set_at(step, pg.Color("red"))



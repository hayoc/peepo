import pygame as pg

from peepo.pp.peepo_network import PeepoNetwork
from peepo.pp.generative_model import GenerativeModel

TRANSPARENT = (0, 0, 0, 0)


class Peepo:
    """
    Abstract class used in generative model.
    Generative Model will call the action and observation functions in order to do active inference and hypothesis
    update respectively.

    When a LEAF node contains 'motor' in the name, the action method will be executed to minimize prediction error.
    Otherwise the LEAF node will minimize the prediction error by fetching the OBSERVED value and performing a
    hypothesis update.
    """

    def __init__(self, name: str, network: PeepoNetwork, graphical: bool, pos=(0, 0), size=(4, 4)):
        self.name = name
        self.network = network
        self.graphical = graphical
        self.rect = pg.Rect(pos, size)
        self.rect.center = pos
        self.rotation = 0
        self.generative_model = GenerativeModel(self, n_jobs=1)
        if self.graphical:
            self.image = self.make_image()
            self.image_original = self.image.copy()

    def action(self, node, prediction):
        pass

    def observation(self, name):
        pass

    def update(self):
        pass

    def draw(self, surface):
        pass

    def make_image(self):
        image = pg.Surface(self.rect.size).convert_alpha()
        image.fill(TRANSPARENT)
        image_rect = image.get_rect()
        pg.draw.rect(image, pg.Color("black"), image_rect)
        pg.draw.rect(image, pg.Color("green"), image_rect.inflate(-2, -2))
        return image

    def clip(self):
        if self.rect.x < 0:
            self.rect.x = 799
        if self.rect.x > 800:
            self.rect.x = 1
        if self.rect.y < 0:
            self.rect.y = 799
        if self.rect.y > 800:
            self.rect.y = 1
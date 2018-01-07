

class PeepoVirtual:
    """
    Virtual Peepo implementation for testing purposes.
    """

    def __init__(self):
        pass

    def vision(self):
        return self.infrared

    def touch(self):
        pass

    def drive(self, speed):
        self.moving[0] = 0.1
        self.moving[1] = 0.9

    def steer(self, degrees):
        self.infrared[0] = 0.1
        self.infrared[1] = 0.9

    def stop(self):
        pass

    def shutdown(self):
        pass

    def set_infrared(self, infrared):
        self.infrared = infrared

    def set_moving(self, moving):
        self.moving = moving

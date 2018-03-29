

class PeepoVirtual:
    """
    Virtual Peepo implementation for testing purposes.
    """

    def __init__(self):
        self.infrared = 120
        self.left_motor = 0
        self.right_motor = 0

    def vision(self):
        return self.infrared

    def obstacle(self, visible):
        self.infrared = 5 if visible else 120

    def forward(self):
        self.left_motor = 500
        self.right_motor = 500

    def backward(self):
        self.left_motor = -500
        self.right_motor = -500

    def is_driving_forward(self):
        return self.left_motor > 250 and self.right_motor > 250

    def is_driving_backward(self):
        return self.left_motor > 250 and self.right_motor > 250

    def turn_left(self):
        pass

    def turn_right(self):
        pass

    def stop(self):
        pass

    def shutdown(self):
        pass

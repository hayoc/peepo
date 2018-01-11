#!/usr/bin/env python3

import ev3dev.ev3 as ev3
import numpy as np
import logging

class Peepo:

    def __init__(self):
        self.steering = ev3.MediumMotor('outB')
        self.driving = ev3.LargeMotor('outA')
        self.touching = ev3.TouchSensor()
        assert self.touching.connected
        self.infrared = ev3.InfraredSensor()
        assert self.infrared.connected
        self.infrared.mode = 'IR-PROX'

    def vision(self):
        value = self.infrared.value()
        logging.info("PEEPO VISION: " + str(value))
        # return [OBSTACLE NO] if val > 30 else [OBSTACLE YES]
        return np.array([0.1, 0.9]) if value < 30 else np.array([0.9, 0.1])

    def touch(self):
        return self.touching.value()

    def movement(self):
        speed = self.driving.count_per_rot
        logging.info("PEEPO SPEED: " + str(speed))
        # return [MOVING NO] if val < 500 else [MOVING YES]
        return np.array([0.9, 0.1]) if speed < 500 else np.array([0.1, 0.9])

    def drive(self, speed):
        logging.info("PEEPO DRIVES")
        self.driving.run_forever(speed_sp=2000 * speed - 1000)
        logging.info("PEEPO STEERING POS: " + str(self.steering.position_sp))
        self.straight()

    def steer(self, degrees):
        logging.info("PEEPO STEERS")
        self.steering.run_to_rel_pos(position_sp=180 * degrees - 90, speed_sp=900, stop_action="brake")

    def straight(self):
        self.steering.run_to_abs_pos(position_sp=0, speed_sp=900, stop_action="brake")

    def stop(self):
        self.driving.stop(stop_action="brake")

    def shutdown(self):
        self.steering.stop()
        self.driving.stop()
        self.steering.run_forever(speed_sp=0)
        self.driving.run_forever(speed_sp=0)
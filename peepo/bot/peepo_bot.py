#!/usr/bin/env python3

import logging

import ev3dev.ev3 as ev3


class PeepoBot:

    def __init__(self):
        self.right_motor = ev3.LargeMotor('outA')
        assert self.right_motor.connected
        self.left_motor = ev3.LargeMotor('outD')
        assert self.left_motor.connected
        self.infrared = ev3.InfraredSensor()
        assert self.infrared.connected
        self.infrared.mode = 'IR-PROX'

        # self.color = ev3.ColorSensor()
        # assert self.color.connected

    def vision(self):
        value = self.infrared.value()
        logging.info('peepo vision: %s', value)
        return value

    def forward(self):
        logging.info('peepo forward')
        self.left_motor.run_forever(speed_sp=-500)
        self.right_motor.run_forever(speed_sp=-500)

    def backward(self):
        logging.info('peepo backward')
        self.left_motor.run_forever(speed_sp=500)
        self.right_motor.run_forever(speed_sp=500)

    def is_driving_forward(self):
        return self.left_motor.count_per_rot > 400 and self.right_motor.count_per_rot > 400

    def is_driving_backward(self):
        logging.info('left motor: %s - right motor: %s', self.left_motor.duty_cycle, self.right_motor.duty_cycle)
        return self.left_motor.duty_cycle > 50 and self.right_motor.duty_cycle > 50

    def turn_left(self):
        logging.info('peepo left')
        self.right_motor.run_to_rel_pos(position_sp=-360, speed_sp=900, stop_action="brake")

    def turn_right(self):
        logging.info('peepo right')
        self.left_motor.run_to_rel_pos(position_sp=-360, speed_sp=900, stop_action="brake")

    def stop(self):
        logging.info('peepo stop')
        self.left_motor.stop(stop_action="brake")
        self.right_motor.stop(stop_action="brake")

    def shutdown(self):
        self.right_motor.stop()
        self.left_motor.stop()
        self.right_motor.run_forever(speed_sp=0)
        self.left_motor.run_forever(speed_sp=0)

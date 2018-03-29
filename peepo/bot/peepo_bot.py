#!/usr/bin/env python3

import ev3dev.ev3 as ev3


class PeepoBot:

    def __init__(self):
        self.right_motor = ev3.LargeMotor('outA')
        self.left_motor = ev3.LargeMotor('outD')
        # self.color = ev3.ColorSensor()
        # assert self.color.connected
        self.infrared = ev3.InfraredSensor()
        assert self.infrared.connected
        self.infrared.mode = 'IR-PROX'

    def vision(self):
        return self.infrared.value()

    def forward(self):
        self.left_motor.run_forever(speed_sp=-500)
        self.right_motor.run_forever(speed_sp=-500)

    def backward(self):
        self.left_motor.run_forever(speed_sp=500)
        self.right_motor.run_forever(speed_sp=500)

    def is_driving_forward(self):
        return self.left_motor.count_per_rot > 250 and self.right_motor.count_per_rot > 250

    def is_driving_backward(self):
        return self.left_motor.count_per_rot > 250 and self.right_motor.count_per_rot > 250

    def turn_left(self):
        self.right_motor.run_to_rel_pos(position_sp=-360, speed_sp=900, stop_action="brake")

    def turn_right(self):
        self.left_motor.run_to_rel_pos(position_sp=-360, speed_sp=900, stop_action="brake")

    def stop(self):
        self.left_motor.stop(stop_action="brake")
        self.right_motor.stop(stop_action="brake")

    def shutdown(self):
        self.right_motor.stop()
        self.left_motor.stop()
        self.right_motor.run_forever(speed_sp=0)
        self.left_motor.run_forever(speed_sp=0)

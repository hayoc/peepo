import pygame as pg

vec = pg.math.Vector2


def end_line(radius, rotation, center):
    center_rotate = vec(radius, 0).rotate(rotation)
    return center_rotate + center


def observation(rect, center, line1, line2, radius):
    #to construct
    return 20,40,False

def collision(rect, center, line1, line2, radius):
    test1 = False
    test2 = False
    if not ((rect.center[0] - center[0]) ** 2 + (rect.center[1] - center[1]) ** 2) <= radius ** 2:
        point = flat_intersection(rect, center, 0)
        test1 = in_sector(center, line1, line2, radius, point)
    test2 = in_sector(center, line1, line2, radius, vec(rect.center))
    test3 = (line1, line2)
    for line in test3:
        coll = flat_intersection(rect, center, line)
        if coll:
            test3 = True
    if test3 != True:
        test3 = False
    if test1 or test2 or test3:
        return True
    else:
        return False


def flat_intersection(rect, center, line1):
    if not line1:
        line1 = vec(rect.center)
    veccenter = line1 - center
    a = rect.top
    b = rect.bottom
    c = rect.left
    d = rect.right
    x = 0
    y = 0
    t = 0
    hor = (a, b)
    ver = (c, d)
    for line in hor:
        if veccenter.y:
            t = (line - center.y) / veccenter.y
            x = center.x + t * veccenter.x
        else:
            if line == a:
                a = False
            else:
                b = False
        if line and 0 <= t <= 1 and rect.left <= x <= rect.right:
            if line == a:
                a = vec(x, line)
            else:
                b = vec(x, line)
        else:
            if line == a:
                a = False
            else:
                b = False
    for line in ver:
        if veccenter.x:
            t = (line - center.x) / veccenter.x
            y = center.y + t * veccenter.y
        else:
            if line == c:
                c = False
            else:
                d = False
        if line and 0 <= t <= 1 and rect.top <= y <= rect.bottom:
            if line == c:
                c = vec(line, y)
            else:
                d = vec(line, y)
        else:
            if line == c:
                c = False
            else:
                d = False
    z = (a, b, c, d)
    try:
        z = [i for i in z if i is not False][0]
    except IndexError:
        z = False
    return z


def in_sector(center, line_1, line_2, radius, pos):
    if not pos:
        return False
    if ((pos[0] - center[0]) ** 2 + (pos[1] - center[1]) ** 2) <= radius ** 2:
        pos_1 = line_1 - center
        rot_1 = pos_1.angle_to(vec(1, 0)) % 360
        pos_2 = line_2 - center
        rot_2 = pos_2.angle_to(vec(1, 0)) % 360
        angle = (pos - center).angle_to(vec(1, 0)) % 360
        difference_0 = rot_1 - rot_2
        difference_1 = rot_1 - angle
        if difference_0 < 0:
            difference_0 += 360
        if difference_1 < 0:
            difference_1 += 360
        if difference_0 >= difference_1:
            return True
    return False

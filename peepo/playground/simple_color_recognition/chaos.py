
import numpy as np
def main():
    max_loops = 10000000
    start = 1/2#0.03236
    r = 3.5
    r_ = np.longdouble(r)
    r__ = np.single(r)
    x_0 = start
    x = start
    start_ = np.longdouble(start)#is platform defined !!!!!
    x_0_ = start_
    x_ = start_
    start__ = np.single(start)
    x_0__ = start__
    x__ = start__
    for i in range(0, max_loops):
        x = r*x_0*(1 - x_0)
        x_0 = x
        x_ = r_*x_0_*(1 - x_0_)
        x_0_ = x_
        x__ = r__*x_0__*(1 - x_0__)
        x_0__ = x__

    print("Normal         precision  :Result after ", max_loops, ' loops : ', x)
    print("np.longdouble  precision  :Result after ", max_loops, ' loops : ', x_)
    print("np.single      precision  :Result after ", max_loops, ' loops : ', x__)


####################################################################################
############################### BEGIN HERE #########################################
####################################################################################

if __name__ == "__main__":
    # logging.basicConfig()
    # logging.getLogger().setLevel(logging.DEBUG)
    main()
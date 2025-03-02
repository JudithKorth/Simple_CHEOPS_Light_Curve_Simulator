from astropy import units as u
from numpy.random import normal

from pytransit import RRModel
from pytransit.orbits import fold, i_from_baew
from numpy import ceil, linspace, ndarray

CHEOPS_ORBIT = 98 * u.min


class LCSim:
    def __init__(self, window_width: float, exp_time: float, white_noise: float):
        self.window_width = window_width * u.h
        self.exp_time = exp_time * u.s
        self.nexp = int(ceil((self.window_width.to(u.s) / self.exp_time).value))
        self.white_noise = white_noise

        hw = self.window_width.to(u.d).value / 2
        self.time = linspace(-hw, hw, self.nexp)

        self.tm = RRModel(small_planet_limit=0.001)
        self.tm.set_data(self.time)

    def __call__(self, k: float, t0: float, p: float, a: float, b: float, e: float, w: float,
                 limb_darkening: list | tuple | ndarray, efficiency: float, shift: float):
        if not (0.0 <= efficiency <= 1):
            raise ValueError("Efficiency must be between 0 and 1.")

        i = i_from_baew(b, a, e, w)
        flux = self.tm.evaluate(k, limb_darkening, t0, p, a, i, e, w) + normal(0.0, self.white_noise, size=self.nexp)

        phase = fold(self.time, CHEOPS_ORBIT.to(u.d).value, shift=-shift) / CHEOPS_ORBIT.to(u.d).value
        mask = abs(phase) > 0.5 * (1 - efficiency)

        return self.time[mask], flux[mask]

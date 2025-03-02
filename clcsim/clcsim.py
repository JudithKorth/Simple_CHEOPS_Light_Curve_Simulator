from astropy import units as u
from numpy.random import normal

from pytransit import RRModel, EclipseModel
from pytransit.models.numba.phasecurves import lambert_phase_function, emission
from pytransit.orbits import fold, i_from_baew
from numpy import ceil, linspace, ndarray, squeeze

CHEOPS_ORBIT = 99 * u.min


class LCSim:
    def __init__(self, window_width: float, exp_time: float, white_noise: float):
        self.window_width = window_width * u.h
        self.exp_time = exp_time * u.s
        self.nexp = int(ceil((self.window_width.to(u.s) / self.exp_time).value))
        self.white_noise = white_noise

        hw = self.window_width.to(u.d).value / 2
        self.time = linspace(-hw, hw, self.nexp)

        self.tm = RRModel(small_planet_limit=0.001)
        self.em = EclipseModel()
        self.tm.set_data(self.time)
        self.em.set_data(self.time)

    def __call__(self, k: float, t0: float, p: float, a: float, b: float, e: float, w: float,
                 limb_darkening: list | tuple | ndarray,
                 geometric_albedo: float = 0.0, fratio_day: float = 0.0, fratio_night: float = 0.0,
                 efficiency: float = 0.8, eff_phase: float = 0.0):
        if not (0.0 <= efficiency <= 1.0):
            raise ValueError("Efficiency must be between 0 and 1.")

        i = i_from_baew(b, a, e, w)
        f_transit = self.tm.evaluate(k, limb_darkening, t0, p, a, i, e, w)
        f_eclipse = self.em.evaluate(k, t0, p, a, i, e, w, multiplicative=True)
        f_reflection = lambert_phase_function(a, k ** 2, geometric_albedo, t0, p, x_is_time=True, x=self.time)
        f_emission = emission(k ** 2, fratio_night, fratio_day, 0.0, t0, p, x_is_time=True, x=self.time)
        f_noise = normal(0.0, self.white_noise, size=self.nexp)
        flux = f_transit + squeeze(f_eclipse * (f_reflection + f_emission)) + f_noise

        # Approximate and remove SAA and Earth crossings
        phase = fold(self.time, CHEOPS_ORBIT.to(u.d).value, shift=-eff_phase) / CHEOPS_ORBIT.to(u.d).value
        mask = abs(phase) > 0.5 * (1 - efficiency)

        return self.time[mask], flux[mask]

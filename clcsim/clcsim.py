from astropy import units as u
from numpy.random import normal

from pytransit import RRModel, EclipseModel
from pytransit.models.numba.phasecurves import lambert_phase_function, emission
from pytransit.orbits import fold, i_from_baew
from numpy import ceil, linspace, ndarray, squeeze

CHEOPS_ORBIT = 99 * u.min


class LCSim:
    def __init__(self, window_width: float, exp_time: float, white_noise: float):
        """Simple CHEOPS Light Curve Simulator

        Simulates transit and phase curve light curves for the CHEOPS mission.

        Attributes
        ----------
        window_width : float
            Observation window width in hours.
        exp_time : float
            Exposure time of each observation in seconds.
        white_noise : float
            White noise level to be included in the model.
        nexp : int
            Number of expected observations calculated based on the window width and
            exposure time.
        time : ndarray
            Array of time values, evenly spaced over the observation window adjusted for the
            total number of exposures.
        tm : RRModel
            Transit model object initialized with a small planet limit of 0.001 and
            setup using the generated time array.
        em : EclipseModel
            Eclipse model object initialized and setup using the generated time array.

        Parameters
        ----------
        window_width : float
            Width of the observation window in hours.
        exp_time : float
            Exposure time per measurement in seconds.
        white_noise : float
            White noise level applied to the measurement.
        """
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

    def __call__(self, radius_ratio: float, zero_epoch: float, period: float, scaled_semi_major_axis: float,
                 impact_parameter: float, eccentricity: float = 0.0, argument_of_periastron: float = 0.0,
                 limb_darkening: list | tuple | ndarray = (0.2, 0.3),
                 geometric_albedo: float = 0.0, fratio_day: float = 0.0, fratio_night: float = 0.0,
                 efficiency: float = 0.8, eff_phase: float = 0.0):
        """Simulate a light curve for a given planet configuration.

        The  method computes individual astrophysical contributions to the overall light
        curve, including the transit, eclipse, reflection, and emission components. It
        includes an additive noise component. The time and flux values are filtered to
        remove data affected by specific orbital considerations (e.g., SAA or Earth
        crossings) based on a given efficiency parameter.

        Parameters
        ----------
        radius_ratio : float
            Planet to star radius ratio.
        zero_epoch : float
            Time of the reference transit center [d].
        period : float
            Orbital period of the planet [d].
        scaled_semi_major_axis : float
            Semi-major axis of the planet divided by the star radius.
        impact_parameter : float
            Impact parameter of the orbit.
        eccentricity : float
            Orbital eccentricity.
        argument_of_periastron : float
            Argument of periastron of the orbit [rad].
        limb_darkening : list | tuple | ndarray
            Coefficients for the limb darkening law.
        geometric_albedo : float, optional
            Geometric albedo of the planet.
        fratio_day : float, optional
            Day-side flux ratio of the emission component.
        fratio_night : float, optional
            Night-side flux ratio of the emission component.
        efficiency : float, optional
            Observing efficiency, with a range [0.0, 1.0].
        eff_phase : float, optional
            Orbital phase shift for efficiency cuts.

        Returns
        -------
        tuple of ndarray
            Tuple containing:
                - Masked array of times that meet the efficiency filtering criteria.
                - Masked flux values corresponding to the selected times.
        """
        if not (0.0 <= efficiency <= 1.0):
            raise ValueError("Efficiency must be between 0 and 1.")

        i = i_from_baew(impact_parameter, scaled_semi_major_axis, eccentricity, argument_of_periastron)
        f_transit = self.tm.evaluate(radius_ratio, limb_darkening, zero_epoch, period, scaled_semi_major_axis, i, eccentricity, argument_of_periastron)
        f_eclipse = self.em.evaluate(radius_ratio, zero_epoch, period, scaled_semi_major_axis, i, eccentricity, argument_of_periastron, multiplicative=True)
        f_reflection = lambert_phase_function(scaled_semi_major_axis, radius_ratio ** 2, geometric_albedo, zero_epoch, period, x_is_time=True, x=self.time)
        f_emission = emission(radius_ratio ** 2, fratio_night, fratio_day, 0.0, zero_epoch, period, x_is_time=True, x=self.time)
        f_noise = normal(0.0, self.white_noise, size=self.nexp)
        flux = f_transit + squeeze(f_eclipse * (f_reflection + f_emission)) + f_noise

        # Approximate and remove SAA and Earth crossings
        phase = fold(self.time, CHEOPS_ORBIT.to(u.d).value, shift=-eff_phase) / CHEOPS_ORBIT.to(u.d).value
        mask = abs(phase) > 0.5 * (1 - efficiency)

        return self.time[mask], flux[mask]

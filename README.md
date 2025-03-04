# Simple CHEOPS Light Curve Simulator

A simple Python package to simulate CHEOPS light curves for transits, eclipses, and phase curves. 

[![Docs](https://readthedocs.org/projects/simple-cheops-light-curve-simulator/badge/)](https://simple-cheops-light-curve-simulator.readthedocs.io)
[![Licence](http://img.shields.io/badge/license-GPLv3-blue.svg?style=flat)](http://www.gnu.org/licenses/gpl-3.0.html)
[![PyPI version](https://badge.fury.io/py/cheopslcs.svg)](https://pypi.org/project/cheopslcs/)

## Installation

Install from PyPI

    pip install cheopslcs

## Usage

    lcs = LCSim(window_width=24*1.5, exp_time=60, white_noise=1e-5)
    
    time, flux = lcs(radius_ratio=0.1, zero_epoch=0.0, period=0.7,
                     scaled_semi_major_axis=4.0, impact_parameter=0.1,
                     eccentricity=0.2, argument_of_periastron=0.23*pi,
                     geometric_albedo=0.5, limb_darkening=[0.2, 0.3],
                     efficiency=0.6, eff_phase=0.2)

![](doc/source/example.svg)

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to enhance the functionality or documentation.
License

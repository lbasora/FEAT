# Fuel Estimation in Air Transportation (FEAT)
This is a fork of the original repository containing Jupyter Notebooks and csv files that support the submitted paper *Fuel Estimation in Air Transportation: Modeling global fuel consumption for commercial aviation*.

In addition, an implementation of the paper algorithms in Python based on the [OpenAP](https://github.com/junzis/openap) is under development. The library can import traffic data generated with the [traffic](https://github.com/xoolive/traffic) library.

## Installation
```sh
git clone https://github.com/lbasora/feat
cd feat
python setup.py install
```

## Citing FEAT

The algorithms are described in the following paper (see Appendix A to K in supplementary information for algorithm details):
```bibtex
@article{seymour_fuel_2020,
	title = {Fuel {Estimation} in {Air} {Transportation}: {Modeling} global fuel consumption for commercial aviation},
	author = {Seymour, K. and Held, M. and Georges, G. and Boulouchos, K.},
	journal = {Transportation Research Part D: Transport and Environment},
	volume = {88},
	year = {2020},
	pages = {102528},
	issn = {13619209},
	doi = {10.1016/j.trd.2020.102528},
}
```

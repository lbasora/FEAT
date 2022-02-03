import numpy as np
import pandas as pd
from openap import aero, FuelFlow, Thrust, prop
from .flight import FlightProfiles, FlightProfileGenerator
from .fleet import FleetData
from tqdm.autonotebook import tqdm


class FuelEstimator:
    def __init__(self, ac_type, eng_type=None):
        self.ac_type = ac_type
        self.eng_type = eng_type
        self.thrust = Thrust(ac=ac_type, eng=eng_type)
        self.fuelflow = FuelFlow(ac=ac_type, eng=eng_type, use_synonym=True)
        self.mass = Mass(self, ac_type=ac_type, eng_type=eng_type)

    def __call__(self, flight_profiles, last_point=False):
        length = len(flight_profiles)

        def generate():
            for fp in tqdm(flight_profiles, desc="Flight profiles", total=length):
                yield self.compute_fuel(fp, self.mass.compute_tow(fp), last_point)

        return FlightProfiles(generate(), length)

    def compute_fuel(self, flight_profile, mass, last_point=False):
        """
        Compute fuel from flight profile:
        Iteration over Dataframe needed:
            Implemented via numpy array iteration (faster than iterrows or itertuples):
            https://towardsdatascience.com/heres-the-most-efficient-way-to-iterate-through-your-pandas-dataframe-4dad88ac92ee
            TODO: cython, numba implementation ?
        """

        def compute_thr(fp, v, h, vs):
            if fp == "TO":
                return self.thrust.takeoff(tas=v, alt=h)
            if fp == "CL":
                return self.thrust.climb(tas=v, alt=h, roc=vs)
            if fp == "CR":
                return self.thrust.cruise(tas=v, alt=h)
            if fp == "DE":
                return self.thrust.descent_idle(tas=v, alt=h)
            return np.NaN

        def compute_ff(fp, v, h, thr, m):
            if fp == "TO":
                return self.fuelflow.takeoff(tas=v, alt=h, throttle=1)
            if fp == "CR":
                return self.fuelflow.enroute(mass=m, tas=v, alt=h)
            if thr == thr:
                return self.fuelflow.at_thrust(acthr=thr, alt=h)
            return np.NaN

        flight_profile = flight_profile.assign(
            thr=np.NaN, ff=np.NaN, fc=np.NaN, m=np.NaN
        )
        t = flight_profile.t.values
        fp = flight_profile.fp.values
        v = flight_profile.v.values
        h = flight_profile.h.values
        vs = flight_profile.vs.values
        thr = flight_profile.thr.values
        ff = flight_profile.ff.values
        fc = flight_profile.fc.values
        m = flight_profile.m.values
        m_prev = mass
        t_prev = 0
        for i in range(len(flight_profile)):
            if thr[i] == np.NaN:
                continue
            v_i, h_i, vs_i = v[i] / aero.kts, h[i] / aero.ft, vs[i] / aero.fpm
            thr[i] = compute_thr(fp[i], v_i, h_i, vs_i)
            ff[i] = compute_ff(fp[i], v_i, h_i, thr[i], m_prev)
            dt = t[i] - t_prev
            t_prev = t[i]
            fc[i] = ff[i] * dt
            m[i] = m_prev - fc[i]
            m_prev = m[i]

        flight_profile = flight_profile.assign(fc=flight_profile.fc.cumsum(skipna=True))
        flight_profile["ff"] = flight_profile["ff"].astype("float")
        if last_point:
            last_point = flight_profile.query("m==m").iloc[-1]
            return pd.DataFrame.from_records(
                [(last_point.id, last_point.s, last_point.fc, last_point.m)],
                columns=["id", "fd", "fc", "m"],
            )
        return flight_profile


class Mass:
    def __init__(self, fe, ac_type, eng_type=None):
        self.aircraft = prop.aircraft(ac_type)
        self.fleet = FleetData(ac_type)
        self.fpg = FlightProfileGenerator(ac_type=ac_type, eng_type=eng_type)
        self.fe = fe

    @property
    def reference_mass(self):
        return (
            self.aircraft["limits"]["OEW"]
            + (self.aircraft["limits"]["MTOW"] - self.aircraft["limits"]["OEW"]) * 0.7
        )

    @property
    def oew(self):
        return self.aircraft["limits"]["OEW"]

    def compute_payload_mass(self):
        """TODO use fleet and updated ICAO/IATA data"""
        avg_num_seats = self.fleet.get_avg_num_seats()

        # ICAO, 2009. Traffic - Commercial Air Carriers Reporting Instructions, Form A.
        # International Civil Aviation Organization. URL http://www.icao.int/staforms.
        passenger_luggage_mass = 100  # KG

        # IATA, 2019. Economic Performance of the Airline Industry, 2019 End-year report.
        # International Air Transport Association.
        # URL https://www.iata.org/contentassets/36695cd211574052b3820044111b56de/airline-industry-economic-performance-dec19-report.pdf.
        avg_load_factor = 0.819  # for 2018

        return avg_load_factor * avg_num_seats * passenger_luggage_mass

    def compute_tow(self, flight_profile, return_tow_only=True):
        """
        See appendix G FEAT paper for algorithm details fuel load estimation:
        https://ars.els-cdn.com/content/image/1-s2.0-S136192092030715X-mmc8.pdf
        """
        tow = self.reference_mass
        res_cruise = self.fpg.gen_cruise_for_fuel_reserve()
        alt_flight = self.fpg.gen_flight_for_alternate_fuel()
        cumul = []
        while True:
            f_trip = self.fe.compute_fuel(
                flight_profile, mass=tow, last_point=True
            ).fc.item()
            f_cont = f_trip * 0.05
            landing_mass = tow - f_trip
            f_res = self.fe.compute_fuel(
                res_cruise, mass=landing_mass, last_point=True
            ).fc.item()
            f_alt = self.fe.compute_fuel(
                alt_flight, mass=landing_mass, last_point=True
            ).fc.item()
            m_fuel = f_trip + f_cont + f_res + f_alt
            new_tow = self.oew + self.compute_payload_mass() + m_fuel
            cumul.append((f_trip, f_cont, f_res, f_alt, m_fuel, tow, new_tow))
            if abs(tow - new_tow) < 10:
                break
            tow = new_tow

        if return_tow_only:
            return tow

        return tow, pd.DataFrame.from_records(
            cumul,
            columns=["f_trip", "f_cont", "f_res", "f_alt", "m_fuel", "tow", "new_tow"],
        )

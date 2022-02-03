import numpy as np
import pandas as pd
from openap import WRAP, aero, FlightPhase
from openap.traj import Generator
from openap import prop

from tqdm.autonotebook import tqdm


class FlightProfiles:
    def __init__(self, fpg, length):
        self.fpg = fpg
        self.length = length

    def __iter__(self):
        return (fp for fp in self.fpg)

    def __len__(self) -> int:
        return self.length

    @classmethod
    def from_df(cls, df):
        dfg = df.groupby("id")
        return cls((fp for _, fp in dfg), max(dfg.ngroup()) + 1)

    @classmethod
    def from_traffic(cls, t):
        """
        Build flight profiles from a Traffic t structure (see traffic library doc):
        https://traffic-viz.github.io/traffic.core.traffic.html?highlight=traffic#traffic.core.Traffic
        """
        cumul = []
        d = dict()
        for flight in t:
            d["t"] = (
                flight.data.timestamp.diff()
                .dt.seconds.fillna(0)
                .astype("int")
                .cumsum()
                .values
            )
            d["h"] = flight.data.altitude.values * aero.ft
            d["s"] = (
                flight.cumulative_distance(
                    compute_gs=False, compute_track=False
                ).data.cumdist.values
                * aero.nm
            )
            d["v"] = (
                abs(
                    (
                        flight.data.TAS
                        if "TAS" in flight.data.columns
                        else flight.data.groundspeed
                    ).values
                )
                * aero.kts
            )
            d["vs"] = flight.data.vertical_rate.values * aero.fpm
            # d["fp"] = flight.data.phase if "phase" in flight.data.columns else None
            d["id"] = flight.data.flight_id
            df = FlightPhaseEstimator()(pd.DataFrame.from_dict(d))
            cumul.append(df)
        return cls.from_df(pd.concat(cumul))

    def to_df(self):
        return pd.concat(self.fpg).reset_index(drop=True)


def gentraj(ac_type, duration, altitude, dt_cr=60, dt_cl=30, dt_de=30):
    trajgen = Generator(ac=ac_type)
    wrap = WRAP(ac=ac_type, use_synonym=True)
    data_cl = trajgen.climb(dt=dt_cl, alt_cr=altitude)
    data_cr = trajgen.cruise(
        dt=dt_cr,
        range_cr=2 * wrap.cruise_range()["maximum"] * 1e3,
        alt_cr=data_cl["h"][-1] / aero.ft,
    )
    idx = np.abs(data_cr["t"] - duration).argmin() + 1

    data_cr = dict(
        (key, value[:idx] if isinstance(value, np.ndarray) else value)
        for key, value in data_cr.items()
    )
    data_de = trajgen.descent(dt=dt_de, alt_cr=data_cr["h"][-1] / aero.ft)
    return {
        "t": np.concatenate(
            [
                data_cl["t"],
                data_cl["t"][-1] + data_cr["t"],
                data_cl["t"][-1] + data_cr["t"][-1] + data_de["t"],
            ]
        ),
        "h": np.concatenate([data_cl["h"], data_cr["h"], data_de["h"]]),
        "s": np.concatenate(
            [
                data_cl["s"],
                data_cl["s"][-1] + data_cr["s"],
                data_cl["s"][-1] + data_cr["s"][-1] + data_de["s"],
            ]
        ),
        "v": np.concatenate([data_cl["v"], data_cr["v"], data_de["v"]]),
        "vs": np.concatenate([data_cl["vs"], data_cr["vs"], data_de["vs"]]),
        "seg": np.concatenate(
            [data_cl["seg"], np.array(["CR"] * len(data_cr["t"])), data_de["seg"]]
        ),
        "fp": np.concatenate(
            [
                np.array(["TO"]),
                np.array(["TO"] * len(np.where(data_cl["seg"] == "TO")[0])),
                np.array(["CL"] * len(np.where(data_cl["seg"] == "IC")[0])),
                np.array(["CL"] * len(np.where(data_cl["seg"] == "PRE-CAS")[0])),
                np.array(["CL"] * len(np.where(data_cl["seg"] == "CAS")[0])),
                np.array(["CL"] * len(np.where(data_cl["seg"] == "MACH")[0])),
                np.array(["CR"] * len(np.where(data_cl["seg"] == "CR")[0])),
                np.array(["CR"] * len(data_cr["t"])),
                np.array(["CR"] * len(np.where(data_de["seg"] == "CR")[0])),
                np.array(["DE"] * (len(data_de["t"]) - 2)),
                np.array(["LD"]),
            ]
        ),
    }


def compute_new_altitude(traj, residual_dist, cruise_altitude):
    max_speed = traj["v"][np.argmax(traj["v"])]
    time_overshoot = residual_dist * 1e3 / max_speed
    roc = traj["vs"][np.where((traj["fp"] == "CL") & (traj["vs"] > 0))]
    min_rocd = roc[np.argmin(roc)]
    excess_al = time_overshoot / 2 * min_rocd
    return cruise_altitude - excess_al


def gen_flight_profile(
    ac_type,
    target_dist,
    cruise_duration=600,
    residual_dist=10,
    dt_cr=60,
    dt_cl=30,
    dt_de=30,
    verbose=False,
):
    """
    See appendix B of FEAT paper for algorithm details flight mission simulation:
    https://ars.els-cdn.com/content/image/1-s2.0-S136192092030715X-mmc8.pdf
    """
    wrap = WRAP(ac=ac_type, use_synonym=True)
    min_cr_alt, max_cr_alt = (
        wrap.cruise_alt()["minimum"] * 1e3 / aero.ft,
        wrap.cruise_alt()["maximum"] * 1e3 / aero.ft,
    )
    min_cr_range, max_cr_range = (
        wrap.cruise_range()["minimum"],
        wrap.cruise_range()["maximum"],
    )
    cruise_altitude = max_cr_alt
    while abs(residual_dist) > 1:
        if cruise_altitude < min_cr_alt:
            if verbose:
                print(
                    "Warning: cruise_altitude < min_cr_alt", cruise_altitude, min_cr_alt
                )
            return None
        traj = gentraj(
            ac_type,
            cruise_duration,
            cruise_altitude,
            dt_cr=dt_cr,
            dt_cl=dt_cl,
            dt_de=dt_de,
        )
        cruise = traj["s"][np.where(traj["fp"] == "CR")]
        cruise_range = (cruise[-1] - cruise[0]) / 1e3
        cruise_v = traj["v"][np.where(traj["fp"] == "CR")][0]
        if cruise_range < min_cr_range:
            if verbose:
                print(
                    "Warning: cruise_range < min_cr_range",
                    cruise_range,
                    min_cr_range,
                    cruise_duration,
                )
            cruise_duration += (min_cr_range - cruise_range) * 1e3 / cruise_v
            continue
        if cruise_range > max_cr_range:
            if verbose:
                print(
                    "Warning: cruise_range > max_cr_range", cruise_range, max_cr_range
                )
            return None
        flight_dist = traj["s"][-1] / 1e3
        residual_dist = flight_dist - target_dist
        if verbose:
            print(
                np.round(cruise_duration / 3600, 2),
                int(cruise_altitude),
                int(cruise_range),
                flight_dist,
                residual_dist,
            )
        if residual_dist > 1:
            cruise_altitude = compute_new_altitude(traj, residual_dist, cruise_altitude)
            if verbose:
                print(residual_dist, "reducing cruise_altitude", cruise_altitude)
        elif residual_dist < -1:
            cruise_duration += -residual_dist * 1e3 / cruise_v
            if verbose:
                print(residual_dist, "increasing cruise_duration", cruise_duration)
    return traj


class FlightProfileGenerator:
    def __init__(self, ac_type, eng_type=None):
        self.ac_type = ac_type
        self.eng_type = eng_type
        self.trajgen = Generator(ac=ac_type, eng=eng_type, use_synonym=True)
        self.wrap = WRAP(ac=ac_type, use_synonym=True)
        self.aircraft = prop.aircraft(ac_type)

    def __call__(self, step=100, dt=10):
        length = int(
            (self.wrap.cruise_range()["maximum"] - self.wrap.cruise_range()["minimum"])
            // step
            + 1
        )

        def generate():
            for i, range_cr in tqdm(
                enumerate(
                    range(
                        int(self.wrap.cruise_range()["minimum"]),
                        int(self.wrap.cruise_range()["maximum"]),
                        step,
                    ),
                ),
                desc="Flight Profiles",
                leave=False,
                total=length,
            ):
                fp = _to_df(
                    self.trajgen.complete(dt=dt, range_cr=range_cr * 1e3, random=True),
                    id=i,
                )
                yield FlightPhaseEstimator()(fp)

        return FlightProfiles(generate(), length)

    def gen_profiles(self, step=100):
        target_dist_min = int(
            self.wrap.cruise_range()["minimum"]
            + self.wrap.cruise_range()["minimum"] * 0.20
        )
        target_dist_max = int(
            self.wrap.cruise_range()["maximum"]
            + self.wrap.cruise_range()["maximum"] * 0.20
        )
        length = int((target_dist_max - target_dist_min) // step + 1)

        def generate():
            for i, target_dist in tqdm(
                enumerate(range(target_dist_min, target_dist_max, step)),
                desc="Flight Profiles",
                leave=False,
                total=length,
            ):
                # print("target_dist", target_dist)
                traj = gen_flight_profile(self.ac_type, target_dist, verbose=False)
                if traj is not None:
                    yield _to_df(traj, id=i)

        return FlightProfiles(generate(), length)

    def gen_cruise_for_fuel_reserve(self):
        """Values according to FEAT appendix G"""
        duration = (
            45 * 60
            if self.aircraft["engine"]["type"] == "turboprop"
            else 30 * 60  # turbofan/pistion
        )
        cruise = self.trajgen.cruise(dt=duration, alt_cr=1500, random=True)
        cruise = dict(
            (key, value[:2] if isinstance(value, np.ndarray) else value)
            for key, value in cruise.items()
        )
        return _to_df(cruise).assign(fp="CR")

    def gen_flight_for_alternate_fuel(self):
        return FlightPhaseEstimator()(
            _to_df(self.trajgen.complete(dt=30, range_cr=0, random=True))
        )


def _to_df(trajgen, id=0):
    fp = pd.DataFrame.from_dict(trajgen).assign(id=id)
    fp["t"] = fp["t"].astype("int64")
    cols = ["h", "s", "v", "vs"]
    fp[cols] = fp[cols].astype("float")
    return fp


class FlightPhaseEstimator:
    def __init__(self):
        self.fpe = FlightPhase()

    def __call__(self, flight_profile):
        ts = flight_profile["t"].values  # timestamp, int, second
        alt = flight_profile["h"].values / aero.ft  # altitude, int, ft
        spd = flight_profile["v"].values / aero.kts  # speed, int, kts
        roc = flight_profile["vs"].values / aero.fpm  # vertical rate, int, ft/min
        self.fpe.set_trajectory(ts, alt, spd, roc)
        labels = self.fpe.phaselabel()
        flight_profile = flight_profile.assign(fp=labels)
        t_cl = flight_profile.query("fp=='CL'").iloc[0].t
        # t_de = fprof.query("fp=='DE'").iloc[-1].t
        # print(t_cl, t_de)
        # take_off = fprof.query(f"fp=='GND' and t < {t_cl}")
        flight_profile.loc[
            (flight_profile.fp == "GND") & (flight_profile.t < t_cl), "fp"
        ] = "TO"
        # landing = fprof.query(f"fp in {['GND','NA']} and t > {t_de}")
        # traj.loc[(fprof.fp == "NA") & (fprof.t > t_de), "fp"] = "L"
        return flight_profile

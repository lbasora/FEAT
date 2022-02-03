import pandas as pd
from pkgutil import get_data
from io import BytesIO


class FleetData:
    """
    To be completed when fleet data available (e.g. Planespotters)
    Seats dataset extracted from https://flightbi.com/data/flights.csv
    """

    def __init__(self, ac_type):
        self.ac_type = ac_type
        self.seats = pd.read_csv(BytesIO(get_data(__name__, f"data/seats.csv")))

    def get_avg_num_seats(self):
        nb_seats = self.seats.query(f"ac_type=='{self.ac_type}'")
        if nb_seats is not None:
            return int(nb_seats.nb_seats.item())
        return None

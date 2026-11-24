"""Model parameters.

Values marked [paper] come from Fang et al. (2026), Section 5.1 and Table 4.
Values marked [assumed] are NOT given in the paper or the instance files and
must be confirmed (see README, "Open assumptions").
"""
from dataclasses import dataclass
import math


@dataclass
class Params:
    # ---- units -----------------------------------------------------------
    dist_scale: float = 0.1        # km per coordinate unit   [calibrated, see README]
    # time unit of the instance files is taken to be minutes           [assumed]

    # ---- drone [paper, Table 4] -------------------------------------------
    speed_kmh: float = 79.2        # 22 m/s
    E: float = 100.0               # battery capacity, Wh
    Pa: float = 43.68              # W per kg
    Pb: float = 113.02             # hover power, W
    m: float = 5.0                 # unloaded mass, kg
    Q: float = 5.0                 # payload capacity, kg

    # ---- costs [paper, Section 5.1] ----------------------------------------
    Ct: float = 1.0                # $ per km flown
    Cp: float = 1.0                # $ per unit of lost satisfaction
    Cfixed: float = 100.0          # $ per drone route

    # ---- time windows -------------------------------------------------------
    tol: float = 30.0              # tolerable window = preferred +/- 0.5 h [paper]

    # ---- fixed stations -----------------------------------------------------
    charge_rate: float = 80.0      # Wh per minute ("80w per minute")   [assumed reading]
    pads: int | None = None        # None = exogenous queue (base paper);
                                   # integer = pads per station, queue caused by own fleet
    exo_lambda: float = 0.05       # external arrivals per minute       [assumed]
    exo_mean_s: float = 0.6        # mean external charging time, min   [assumed]
    exo_var_s: float = 0.1         # its variance                       [assumed]
    live_info: bool = True         # policy sees live pad occupancy (else avg_wait_est)
    avg_wait_est: float = 0.0      # wait assumed by the policy when live_info is False

    # ---- UGVs ---------------------------------------------------------------
    n_ugv: int = 1                 # base paper: one UGV
    Ts: float = 3.0                # battery swap time, min             [paper]
    ugv_speed_kmh: float = 54.0    # 15 m/s (paper: random in 10..20)   [assumed midpoint]
    ugv_kwh_per_km: float = 1.0    # used only inside the regret rule   [paper]
    ugv_stock: float = math.inf    # swap batteries per UGV (inf = base paper)
    ugv_range_km: float = math.inf # travel budget per UGV (inf = base paper)
    Cugv_km: float = 0.0           # $ per UGV km        (0 = base objective, Eq. 14)
    Cugv_swap: float = 0.0         # $ per swap
    Cugv_use: float = 0.0          # $ per UGV used

    # ---- regret / cost rule weights [paper] ----------------------------------
    xi1: float = 0.5
    xi2: float = 0.5

    # ---- charging decision ----------------------------------------------------
    filter_late: bool = True       # never pick an option that makes the next
                                   # customer late when another option avoids it

    # ---- search ---------------------------------------------------------------
    infeasible_penalty: float = 1000.0  # added per violated constraint

    # ---- derived ----------------------------------------------------------------
    @property
    def exo_wait(self) -> float:
        """Expected M/G/1 waiting time for the exogenous queue (paper Eq. 31)."""
        rho = self.exo_lambda * self.exo_mean_s
        if rho >= 1.0:
            return math.inf
        es2 = self.exo_var_s + self.exo_mean_s ** 2
        return self.exo_lambda * es2 / (2.0 * (1.0 - rho))

    def km(self, d_units: float) -> float:
        return d_units * self.dist_scale

    def fly_min(self, d_units: float) -> float:
        return self.km(d_units) / self.speed_kmh * 60.0

    def fly_wh(self, d_units: float, load: float) -> float:
        """Paper Eq. 33: e = d / v * (Pa * (m + q) + Pb)."""
        return self.km(d_units) / self.speed_kmh * (self.Pa * (self.m + load) + self.Pb)

    def ugv_min(self, d_units: float) -> float:
        return self.km(d_units) / self.ugv_speed_kmh * 60.0

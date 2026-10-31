from .common import cluster_metric, initial_states, simulate
from .gradient import Gradient
from .mitrano2019 import Mitrano2019
from .rezeck2021 import Rezeck2021
from .santos2020 import Santos2020

CONTROLLERS = {"santos2020": Santos2020, "mitrano2019": Mitrano2019,
               "rezeck2021": Rezeck2021, "gradient": Gradient}
LABELS = {"santos2020": "Santos et al. (2020)", "mitrano2019": "Mitrano et al. (2019)",
          "inacio2019": "In\u00e1cio et al. (2019)", "rezeck2021": "Rezeck et al. (2021)",
          "gradient": "Gradient baseline"}

try:
    from .inacio2019 import Inacio2019
    import rvo2
    CONTROLLERS["inacio2019"] = Inacio2019
except ImportError:
    pass


def make(name, robots, groups, world, seed, cfg=None):
    if name not in CONTROLLERS:
        raise KeyError(f"controller {name!r} unavailable (inacio2019 needs rvo2)")
    kw = {}
    if name == "santos2020" and cfg is not None:
        kw["velocity_sign"] = cfg["controllers"]["santos_velocity_sign"]
    return CONTROLLERS[name](robots, groups, world, seed, **kw)

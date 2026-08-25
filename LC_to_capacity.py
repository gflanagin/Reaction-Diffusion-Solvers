import numpy as np

def land_cover_to_carrying_capacity(land_cover_array, K_base=2.0):
    """
    Map NLCD class codes to carrying capacity values.
    K_base is the baseline carrying capacity for the most hospitable environment.
    """
    lc_map = {
        0:   K_base,       # unknown/nodata
        11:  0,          # Open water -- no population
        12:  0.1*K_base,   # Perennial ice/snow
        21:  0.8*K_base,   # Developed open space
        22:  0.6*K_base,   # Developed low intensity
        23:  0.4*K_base,   # Developed medium intensity
        24:  0.2*K_base,   # Developed high intensity
        31:  0.4*K_base,   # Barren rock
        41:  0.9*K_base,   # Deciduous forest
        42:  0.8*K_base,   # Evergreen forest
        43:  0.85*K_base,  # Mixed forest
        52:  0.75*K_base,   # Shrub/scrub
        71:  K_base,       # Grassland/herbaceous
        81:  0.9*K_base,   # Pasture/hay
        82:  0.8*K_base,   # Cultivated crops
        90:  0.35*K_base,   # Woody wetlands
        95:  0.5*K_base,   # Emergent herbaceous wetlands
    }
    result = np.full(len(land_cover_array), float(K_base), dtype=np.float64)
    for code, value in lc_map.items():
        result[land_cover_array == code] = value
    return result
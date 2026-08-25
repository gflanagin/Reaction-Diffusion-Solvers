import json
import numpy as np

# Land cover metadata -- name, color (RGB 0-1), diffusivity, carrying capacity
lc_metadata = {
    11: {"name": "Open Water",                  "color": [0,0,1], "diffusivity": 0.0001,  "K_fraction": 0.0},
    12: {"name": "Perennial Ice/Snow",           "color": [1,1,1], "diffusivity": 0.3,   "K_fraction": 0.1},
    21: {"name": "Developed Open Space",         "color": [192/255, 192/255, 192/255], "diffusivity": 0.7,   "K_fraction": 0.8},
    22: {"name": "Developed Low Intensity",      "color": [160/255, 160/255, 160/255], "diffusivity": 0.4,   "K_fraction": 0.6},
    23: {"name": "Developed Medium Intensity",   "color": [128/255, 128/255, 128/255], "diffusivity": 0.2,   "K_fraction": 0.4},
    24: {"name": "Developed High Intensity",     "color": [96/255, 96/255, 96/255], "diffusivity": 0.05,  "K_fraction": 0.2},
    31: {"name": "Barren Rock",                  "color": [1, 102/255, 102/255], "diffusivity": 0.9,   "K_fraction": 0.4},
    41: {"name": "Deciduous Forest",             "color": [0, 204/255, 0], "diffusivity": 0.5,   "K_fraction": 0.9},
    42: {"name": "Evergreen Forest",             "color": [0, 153/255, 0], "diffusivity": 0.4,   "K_fraction": 0.8},
    43: {"name": "Mixed Forest",                 "color": [0, 102/255, 0], "diffusivity": 0.45,  "K_fraction": 0.85},
    52: {"name": "Shrub/Scrub",                  "color": [1, 1, 102/255], "diffusivity": 0.8,   "K_fraction": 0.75},
    71: {"name": "Grassland/Herbaceous",         "color": [178/255, 1, 102/255], "diffusivity": 0.95,  "K_fraction": 1.0},
    81: {"name": "Pasture/Hay",                  "color": [1,1, 153/255], "diffusivity": 0.85,  "K_fraction": 0.9},
    82: {"name": "Cultivated Crops",             "color": [1, 102/255, 1], "diffusivity": 0.8,   "K_fraction": 0.8},
    90: {"name": "Woody Wetlands",               "color": [153/255, 1, 204/255], "diffusivity": 0.2,   "K_fraction": 0.35},
    95: {"name": "Emergent Herbaceous Wetlands", "color": [153/255, 1, 1], "diffusivity": 0.3,   "K_fraction": 0.5},
}

opacities = [.6] * len(lc_metadata)

def save_paraview_colormap(lc_metadata, output_json, K_base=2.0):
    """Save a ParaView categorical colormap JSON file."""
    
    annotations = []
    colors = []
    
    for code, meta in lc_metadata.items():
        label = (f"{meta['name']} "
                 f"(D={meta['diffusivity']:.2f}, "
                 f"K={meta['K_fraction']*K_base:.2f})")
        annotations.append(str(float(code)))
        annotations.append(label)
        colors.extend(meta["color"])

    colormap = [
        {
            "ColorSpace": "RGB",
            "Creator": "SIR Model LC Colormap",
            "Name": "NLCD Land Cover",
            "NanColor": [0.5, 0.5, 0.5],
            "RGBPoints": [],
            "Annotations": annotations,
            "IndexedColors": colors,
            "IndexedOpacities": opacities,
            "Discretize": True,
            "NumberOfTableValues": len(lc_metadata),
            "InterpretValuesAsCategories": True,
        }
    ]
    
    with open(output_json, 'w') as f:
        json.dump(colormap, f, indent=2)
    
    print(f"Colormap saved to {output_json}")
    print(f"\nLand Cover Summary (K_base={K_base}):")
    print(f"{'Code':<6} {'Name':<35} {'Diffusivity':<14} {'K':<8}")
    print("-" * 65)
    for code, meta in lc_metadata.items():
        print(f"{code:<6} {meta['name']:<35} {meta['diffusivity']:<14.2f} {meta['K_fraction']*K_base:<8.2f}")

land_cover_classes = np.load("land_cover_classes.npy").astype(np.int32)

save_paraview_colormap(lc_metadata, "nlcd_colormap.json", K_base=2.0)
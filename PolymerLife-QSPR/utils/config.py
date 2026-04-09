PIPELINE_CONFIG = {
    "x": {
        "col_name": "SMILES",
    },

    "y1": {
        "col_name": "Tc",
        "display_name": "Thermal_Conductivity",
        "units": "W/(m*K)"
    },

    "y2": {
        "col_name": "durability_retention",
        "display_name": "Durability_Retention",
        "units": "%"
    },

    "env_features": ['T_env', 'humidity', 'exposure_hours']
}
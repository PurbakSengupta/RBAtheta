"""The rbaTheta model"""
#from giddy.markov import LISA_Markov, Spatial_Markov
#from libpysal.weights import Queen, DistanceBand
#import libpysal
#import geopandas
import pandas as pd
import numpy as np
import core.helpers as fn
import core.event_extraction as ee

def calculate_adaptive_threshold(data, season, hour):
    """
    Calculate adaptive threshold based on season and hour.
    Args:
        data: Input time series data.
        season: Season (e.g., "winter", "summer").
        hour: Hour of the day (0-23).
    Returns:
        Adaptive threshold value.
    """
    # Define seasonal and hourly scaling factors
    seasonal_factors = {
        "winter": 0.01,  # Higher threshold for winter
        "summer": 0.003,  # Lower threshold for summer
        "default": 0.05   # Default threshold for other seasons
    }
    hourly_factors = {
        "day": 0.01,      # Higher threshold during the day (6 AM - 6 PM)
        "night": 0.003     # Lower threshold during the night (6 PM - 6 AM)
    }

    # Determine the scaling factor based on season
    season_factor = seasonal_factors.get(season, seasonal_factors["default"])

    # Determine the scaling factor based on hour
    if 6 <= hour < 18:  # Daytime hours
        hour_factor = hourly_factors["day"]
    else:  # Nighttime hours
        hour_factor = hourly_factors["night"]

    # Calculate adaptive threshold
    mean = np.mean(data)
    std = np.std(data)
    adaptive_threshold = mean + (season_factor * hour_factor) * std
    print(f"Adaptive Threshold: {adaptive_threshold}")

    return adaptive_threshold

def RBA_theta(data, nominal, s=0.01, k=3, fc=0.3):
    """
    Args:
        data: Discrete wind power (/hour) in Watts, from N turbines.
        nominal: Nominal production in Watts.
        s: Smoothness factor of Bspline.
        k: Degree of Bspline.
        fc: Cutoff frequency for Blackman filter.
    Returns:
        Dictionary of [dataframe of events per turbines] x turbines.
    """
    # Ensure the data has a DateTime index
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data must have a DateTime index.")

    N = len(data.columns)
    turbines = [f'Turbine_{i}' for i in range(1, N + 1)]
    normalized_data = {}

    for i in range(N):
        normalized_data[turbines[i]] = fn.normalize(data=data.iloc[:, i].values, nominal=nominal)
        print(f"Normalized Data for {turbines[i]}: {normalized_data[turbines[i]]}")

    tao = len(normalized_data) + 1
    significant_events, stationary_events = {}, {}

    for i in range(N):
        # Extract timestamp from the data (assuming the index is DateTime)
        timestamp = data.index[i]  # Assuming the index is DateTime
        season = get_season(timestamp)  # Get season from timestamp
        hour = timestamp.hour  # Get hour from timestamp

        # Calculate adaptive threshold
        adaptive_threshold = calculate_adaptive_threshold(normalized_data[turbines[i]], season, hour)

        # Detect significant and stationary events using the adaptive threshold
        significant_events[turbines[i]] = ee.significant_events(data=normalized_data[turbines[i]], threshold=adaptive_threshold)
        stationary_events[turbines[i]] = ee.stationary_events(data=normalized_data[turbines[i]], threshold=adaptive_threshold)

    return [significant_events, stationary_events, tao]

def get_season(timestamp):
    """
    Determine season from a timestamp.
    Args:
        timestamp: Datetime object or string.
    Returns:
        season: Season (e.g., "winter", "summer").
    """
    month = timestamp.month
    if month in [12, 1, 2]:
        return "winter"
    elif month in [6, 7, 8]:
        return "summer"
    else:
        return "default"
    '''
    for i in range(N):
        number_of_significant_events = len(significant_events[turbines[i]])
        number_of_stationary_events = len(stationary_events[turbines[i]])

        # initializing the rainflow counts
        #significant_events[turbines[i]]['φ_m'] = [0 * len(significant_events[turbines[i]])]
        #stationary_events[turbines[i]]['φ_s'] = [0 * len(stationary_events[turbines[i]])]

        for k in range(number_of_significant_events):
            start = int(significant_events[turbines[i]].loc[k, 't1'])
            end = int(significant_events[turbines[i]].loc[k, 't2'])
            significant_events[turbines[i]].loc[k, 'φ_m'] = fn.rainflow_count(data=data.iloc[i, start:end])

        for k in range(number_of_stationary_events):
            start = int(stationary_events[turbines[i]].loc[k, 't1'])
            end = int(stationary_events[turbines[i]].loc[k, 't2'])
            stationary_events[turbines[i]].loc[k, 'φ_s'] = fn.rainflow_count(data=data.iloc[i, start:end])
    '''



def markov(major, stationary, shp_path):

    """
    A commonly-used type of weights is Queen-contiguity weights, which reflects adjacency relationships as a binary
    indicator variable denoting whether or not a polygon shares an edge or a vertex with another polygon. These weights
    are symmetric.
    """

    df = geopandas.read_file(shp_path)
    points = [(poly.centroid.x, poly.centroid.y) for poly in df.geometry]
    radius_km = libpysal.cg.sphere.RADIUS_EARTH_KM
    threshold = libpysal.weights.min_threshold_dist_from_shapefile(shp_path, radius=radius_km)
    distance_weights = DistanceBand(points, threshold=threshold*.025, binary=False)
    transition_matrises = {}


    #for lisa markov
    transition_matrises['∆t_m_tran'] = LISA_Markov(major['∆t_m'], distance_weights)
    transition_matrises['∆w_m_tran'] = LISA_Markov(major['∆w_m'], distance_weights)
    transition_matrises['θ_m_tran'] = LISA_Markov(major['θ_m'], distance_weights)
    transition_matrises['σ_m_tran'] = LISA_Markov(major['σ_m'], distance_weights)

    transition_matrises['∆t_s_tran'] = LISA_Markov(stationary['∆t_s'], distance_weights)
    transition_matrises['σ_s_tran'] = LISA_Markov(stationary['σ_s'], distance_weights)

    '''#for spatial markov
    transition_matrises['∆t_m_tran'] = Spatial_Markov(major['∆t_m'], distance_weights, fixed=True, k=5, m=5)
    transition_matrises['∆w_m_tran'] = Spatial_Markov(major['∆w_m'], distance_weights, fixed=True, k=5, m=5)
    transition_matrises['θ_m_tran'] = Spatial_Markov(major['θ_m'], distance_weights, fixed=True, k=5, m=5)
    transition_matrises['σ_m_tran'] = Spatial_Markov(major['σ_m'], distance_weights, fixed=True, k=5, m=5)

    transition_matrises['∆t_s_tran'] = Spatial_Markov(stationary['∆t_s'], distance_weights, fixed=True, k=5, m=5)
    transition_matrises['σ_s_tran'] = Spatial_Markov(stationary['σ_s'], distance_weights, fixed=True, k=5, m=5)'''
    return transition_matrises


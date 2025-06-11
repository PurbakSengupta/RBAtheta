"""The rbaTheta model"""
#from giddy.markov import LISA_Markov, Spatial_Markov
#from libpysal.weights import Queen, DistanceBand
#import libpysal
#import geopandas
import pandas as pd
import numpy as np
import core.helpers as fn
import core.event_extraction as ee
from core.database import RBAThetaDB

def likelihood(threshold, data):
    """Defines the likelihood function for selecting an optimal threshold."""
    event_count = np.sum(np.abs(data) > threshold)  # Number of detected events
    if event_count == 0:
        return 1e-10  # Avoid zero probability
    return event_count  # Higher event count = better likelihood

def propose_new_threshold(current_threshold, step_size=0.01):
    """Proposes a new threshold by adding a small random step."""
    return current_threshold + np.random.uniform(-step_size, step_size)

def calculate_adaptive_threshold(data):
    """
    Calculate adaptive threshold based on the statistical properties of the data.
    Args:
        data: Input time series data.
    Returns:
        Adaptive threshold value.
    """
    mean = np.mean(data)
    std = np.std(data)
    adaptive_threshold = mean + (0.05 * std)  # Adjust the multiplier as needed
    print(f"Adaptive Threshold: {adaptive_threshold}")
    return adaptive_threshold

def calculate_adaptive_threshold_mcmc(data, iterations=5000, initial_threshold=0.4, step_size=0.01):
    """
    Uses MCMC to determine an adaptive threshold while preventing extreme values.
    """
    current_threshold = initial_threshold
    best_threshold = current_threshold
    best_likelihood = likelihood(current_threshold, data)

    for i in range(iterations):
        new_threshold = propose_new_threshold(current_threshold, step_size)

        # Ensure the threshold remains within a valid range
        new_threshold = max(0.01, min(1.0, new_threshold))

        # Compute new likelihood and acceptance probability
        new_likelihood = likelihood(new_threshold, data)
        acceptance_prob = min(1, new_likelihood / (best_likelihood + 1e-5))  # Avoid division errors

        # Accept or reject based on probability
        if np.random.rand() < acceptance_prob:
            current_threshold = new_threshold
            best_likelihood = new_likelihood
            best_threshold = new_threshold

        # Adaptive Step Size: Reduce after 50% iterations
        if i > iterations // 2:
            step_size *= 0.99  # Gradually refine threshold search

    print(f"Fixed MCMC Adaptive Threshold: {best_threshold}")
    return best_threshold

def RBA_theta(data, nominal, s=0.01, k=3, fc=0.3, db_path="rbatheta.db"):
    """
    Args:
        data: Discrete wind power (/hour) in Watts, from N turbines.
        nominal: Nominal production in Watts.
        s: Smoothness factor of Bspline.
        k: Degree of Bspline.
        fc: Cutoff frequency for Blackman filter.
        db_path: Path to SQLite database file.
    Returns:
        Dictionary of [dataframe of events per turbines] x turbines.
    """
    # Initialize SQLite database
    with RBAThetaDB(db_path) as db:
        # Load and normalize data
        db.load_data(data)
        db.normalize_data(nominal)
        
        # Get all turbine IDs
        turbine_ids = db.get_all_turbine_ids()
        tao = len(turbine_ids) + 1
        
        # Initialize event storage
        significant_events_traditional, stationary_events_traditional = {}, {}
        significant_events_mcmc, stationary_events_mcmc = {}, {}

        # Detect events for each turbine
        for turbine_id in turbine_ids:
            # Get normalized data for this turbine
            turbine_data = db.get_turbine_data(turbine_id)
            data_values = turbine_data['normalized_value'].values
            
            # Traditional threshold detection
            adaptive_threshold = calculate_adaptive_threshold(data_values)
            stationary_threshold = adaptive_threshold * 0.2
            
            sig_events_trad = ee.significant_events(
                data=data_values, 
                threshold=adaptive_threshold, 
                use_mcmc=False
            )
            stat_events_trad = ee.stationary_events(
                data=data_values, 
                threshold=stationary_threshold, 
                use_mcmc=False
            )
            
            significant_events_traditional[turbine_id] = sig_events_trad
            stationary_events_traditional[turbine_id] = stat_events_trad
            
            # MCMC threshold detection
            mcmc_threshold = calculate_adaptive_threshold_mcmc(data_values)
            stationary_threshold_mcmc = mcmc_threshold * 0.2
            
            sig_events_mcmc = ee.significant_events(
                data=data_values, 
                threshold=mcmc_threshold, 
                use_mcmc=True, 
                mcmc_threshold=mcmc_threshold
            )
            stat_events_mcmc = ee.stationary_events(
                data=data_values, 
                threshold=stationary_threshold_mcmc, 
                use_mcmc=True, 
                mcmc_threshold=stationary_threshold_mcmc
            )
            
            significant_events_mcmc[turbine_id] = sig_events_mcmc
            stationary_events_mcmc[turbine_id] = stat_events_mcmc

            # Print debug information
            print(f"\nTurbine: {turbine_id}")
            print(f"Traditional Significant Events: {len(sig_events_trad)}")
            print(f"Traditional Stationary Events: {len(stat_events_trad)}")
            print(f"MCMC Significant Events: {len(sig_events_mcmc)}")
            print(f"MCMC Stationary Events: {len(stat_events_mcmc)}")

            # Save events to database
            db.save_events({turbine_id: sig_events_trad}, 'significant_traditional')
            db.save_events({turbine_id: stat_events_trad}, 'stationary_traditional')
            db.save_events({turbine_id: sig_events_mcmc}, 'significant_mcmc')
            db.save_events({turbine_id: stat_events_mcmc}, 'stationary_mcmc')

        def convert_to_dataframe(event_dict):
            if not event_dict:  # If empty
                return pd.DataFrame()
            return pd.concat(event_dict.values(), keys=event_dict.keys()) if isinstance(next(iter(event_dict.values())), pd.DataFrame) else pd.DataFrame(event_dict)

        return [
            convert_to_dataframe(significant_events_traditional),
            convert_to_dataframe(stationary_events_traditional),
            convert_to_dataframe(significant_events_mcmc),
            convert_to_dataframe(stationary_events_mcmc),
            tao
        ]

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


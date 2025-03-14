import time
import os
import multiprocessing
import pandas as pd
from core import save_xls
from core import RBA_theta

os.chdir('.')
BASE_DIR = os.getcwd()
path = os.path.join(BASE_DIR, r'input_data/cleaned_data.xlsx')


def the_test(path):
    """
    Function to test the RBATheta algorithm.
    Args:
        path: Path to the input data file.
    """
    wind_data = pd.read_excel(path)
    wind_data = wind_data.iloc[:100, :]  # Use only the first 100 rows for testing

    # Set DateTime as the index (assuming the first column is DateTime)
    wind_data['DateTime'] = pd.to_datetime(wind_data['DateTime'])
    wind_data.set_index('DateTime', inplace=True)

    # Define nominal power (mean power output from EDA)
    nominal = max(wind_data.max())  # Replace with the actual nominal power if different

    # Run RBATheta with adaptive threshold
    [significant_events, stationary_events, tao] = RBA_theta(data=wind_data, nominal=nominal)

    save_xls(significant_events,
             f'simulations/test_results/all_events/significant_events_T_adaptive.xlsx')
    save_xls(stationary_events,
             f'simulations/test_results/all_events/stationary_events_T_adaptive.xlsx')

def multi_proc(path):
    """
    Creates multiple processes to run the_test function.
    Args:
        path: Path to the input data file.
    """
    processes = [multiprocessing.Process(target=the_test, args=(path,)) for i in range(3)]
    start_time = time.time()
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    delta_time = time.time() - start_time
    print(f'It takes {delta_time:.1f} seconds to run this test')

def main(path):
    """
    Main function to call the multiprocessing function.
    Args:
        path: Path to the input data file.
    """
    multi_proc(path)

if __name__ == '__main__':
    main(path)
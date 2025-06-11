import time
import os
import multiprocessing
import pandas as pd
from core import save_xls
from core import RBA_theta
#from wavelet_processing import calculate_wavelet_threshold

os.chdir('.')
BASE_DIR = os.getcwd()
path = os.path.join(BASE_DIR, r'input_data/fall_data.xlsx')

def the_test(path, traditional_threshold=None):
    # Read and prepare the data
    wind_data = pd.read_excel(path)
    #wind_data = wind_data.iloc[:100, :]  # comment this line for full dataset
    wind_data['DateTime'] = pd.to_datetime(wind_data['DateTime'])
    wind_data.set_index('DateTime', inplace=True)
    nominal = max(wind_data.max())

    # Run RBATheta with SQLite backend
    results = RBA_theta(data=wind_data, nominal=nominal, db_path="wind_data.db")

    # Unpack results
    (sig_events_traditional, stat_events_traditional,
     sig_events_mcmc, stat_events_mcmc,
     tao) = results

    # Verify data types before saving for all methods
    for name, df in [('Traditional_Significant', sig_events_traditional),
                    ('Traditional_Stationary', stat_events_traditional),
                    ('MCMC_Significant', sig_events_mcmc),
                    ('MCMC_Stationary', stat_events_mcmc)]:
        if not isinstance(df, pd.DataFrame):
            print(f"Converting {name} to DataFrame")
            df = pd.DataFrame(df) if df else pd.DataFrame()

    # Create output directory if it doesn't exist
    output_dir = 'simulations/test_results'
    os.makedirs(output_dir, exist_ok=True)

    # Save results
    save_xls({'Traditional_Significant_Events': sig_events_traditional},
             os.path.join(output_dir, 'traditional_significant_events.xlsx'))
    save_xls({'Traditional_Stationary_Events': stat_events_traditional},
             os.path.join(output_dir, 'traditional_stationary_events.xlsx'))
    save_xls({'MCMC_Significant_Events': sig_events_mcmc},
             os.path.join(output_dir, 'mcmc_significant_events.xlsx'))
    save_xls({'MCMC_Stationary_Events': stat_events_mcmc},
             os.path.join(output_dir, 'mcmc_stationary_events.xlsx'))

    print("Event detection completed. Results saved.")

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
    the_test(path)
    #multi_proc(path)

if __name__ == '__main__':
    main(path)
"""
database.py - SQLite database operations for RBATheta
"""
import sqlite3
import pandas as pd
from typing import Union, Optional, Dict, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RBAThetaDB:
    """SQLite database handler for RBATheta model"""
    
    def __init__(self, db_path: str = ":memory:"):
        """
        Initialize the database connection.
        
        Args:
            db_path: Path to SQLite database file (default: in-memory)
        """
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._create_tables()  
        self.cursor.execute("PRAGMA table_info(events)")
        print("Events table columns:", [col[1] for col in self.cursor.fetchall()])
        
    def _create_tables(self):
        """Create tables with all required columns"""
        # Wind data table
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS wind_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            DateTime DATETIME NOT NULL,
            turbine_id TEXT NOT NULL,
            value REAL NOT NULL,
            normalized_value REAL,
            processed BOOLEAN DEFAULT 0
        )
        """)
        
        # Events table - with ALL required columns
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            turbine_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            t1 INTEGER NOT NULL,
            t2 INTEGER NOT NULL,
            delta_t REAL NOT NULL,
            w_t1 REAL NOT NULL,
            w_t2 REAL NOT NULL,
            delta_w REAL NOT NULL,
            sigma REAL NOT NULL,
            theta REAL,
            threshold REAL NOT NULL
        )
        """)
        self.conn.commit()
    
    def load_data(self, data: Union[pd.DataFrame, str], turbine_prefix: str = "Turbine_"):
        """Load data ensuring column name consistency"""
        if isinstance(data, str):
            data = pd.read_excel(data)
        
        # Ensure consistent column naming
        data = data.rename(columns={
            'timestamp': 'DateTime',
            'time': 'DateTime',
            'date': 'DateTime'
        }, errors='ignore')
        
        # Reshape to long format
        data_long = data.reset_index().melt(
            id_vars='DateTime',
            var_name='turbine_id',
            value_name='value'
        )
        
        # Insert with explicit column names
        data_long[['DateTime', 'turbine_id', 'value']].to_sql(
            'wind_data',
            self.conn,
            if_exists='replace',
            index=False
        )
    
    def get_turbine_data(self, turbine_id: str) -> pd.DataFrame:
        """
        Retrieve normalized data for a specific turbine.
        
        Args:
            turbine_id: ID of the turbine to retrieve
            
        Returns:
            DataFrame with timestamp and normalized values
        """
        query = """
        SELECT DateTime, normalized_value 
        FROM wind_data 
        WHERE turbine_id = ? 
        ORDER BY DateTime
        """
        df = pd.read_sql(query, self.conn, params=(turbine_id,), parse_dates=['DateTime'])
        return df.set_index('DateTime')
    
    def get_all_turbine_ids(self) -> List[str]:
        """Get list of all turbine IDs in the database"""
        query = "SELECT DISTINCT turbine_id FROM wind_data"
        return [x[0] for x in self.cursor.execute(query).fetchall()]
    
    def normalize_data(self, nominal: float):
        """
        Normalize all data using the given nominal value.
        
        Args:
            nominal: Nominal production value for normalization
        """
        # First ensure the column exists
        self.cursor.execute("PRAGMA table_info(wind_data)")
        columns = [col[1] for col in self.cursor.fetchall()]
        
        if 'normalized_value' not in columns:
            self.cursor.execute("ALTER TABLE wind_data ADD COLUMN normalized_value REAL")
        
        # Now perform the normalization
        self.cursor.execute("""
        UPDATE wind_data 
        SET normalized_value = value / ?
        WHERE value IS NOT NULL
        """, (nominal,))
        self.conn.commit()
        logger.info(f"Normalized all data with nominal value: {nominal}")
    
    def save_events(self, events: Dict[str, pd.DataFrame], event_type: str):
        """Save events with proper column mapping"""
        for turbine_id, df in events.items():
            # Prepare records with fallback values
            records = []
            for _, row in df.iterrows():
                record = (
                    turbine_id,
                    event_type,
                    int(row.get('t1', 0)),
                    int(row.get('t2', 0)),
                    float(row.get('∆t_m', row.get('∆t_s', 0))),
                    float(row.get('w_m(t1)', row.get('σ_s', 0))),
                    float(row.get('w_m(t2)', row.get('σ_s', 0))),
                    float(row.get('∆w_m', 0)),
                    float(row.get('σ_m', row.get('σ_s', 0))),
                    float(row.get('θ_m', None)) if pd.notna(row.get('θ_m', None)) else None,
                    float(row.get('threshold', 0))
                )
                records.append(record)
            
            # Execute with error handling
            try:
                self.cursor.executemany("""
                INSERT INTO events (
                    turbine_id, event_type, t1, t2, delta_t,
                    w_t1, w_t2, delta_w, sigma, theta, threshold
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, records)
                self.conn.commit()
            except sqlite3.Error as e:
                print(f"Database error: {e}")
                print("Current table schema:")
                self.cursor.execute("PRAGMA table_info(events)")
                print(self.cursor.fetchall())
                raise
    
    def get_events(self, turbine_id: str, event_type: str) -> pd.DataFrame:
        """
        Retrieve events for a specific turbine and event type.
        
        Args:
            turbine_id: ID of the turbine
            event_type: 'significant' or 'stationary'
            
        Returns:
            DataFrame of events
        """
        query = """
        SELECT * FROM events 
        WHERE turbine_id = ? AND event_type = ?
        ORDER BY start_time
        """
        return pd.read_sql(query, self.conn, params=(turbine_id, event_type), parse_dates=['start_time', 'end_time'])
    
    def close(self):
        """Close the database connection"""
        self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
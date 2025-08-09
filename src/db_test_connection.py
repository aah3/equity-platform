from db_manager import DatabaseManager, DatabaseConfig
import dotenv
import os

# Load environment variables
dotenv.load_dotenv()

# Create database configuration
db_config = DatabaseConfig(
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", "5432")),
    username=os.getenv("DB_USERNAME"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_NAME"),
    schema=os.getenv("DB_SCHEMA", "public")
)

# Create database manager
db_manager = DatabaseManager(db_config)

# Test connection with a simple query
try:
    result = db_manager.execute_query("SELECT 1 as test;")
    print("Connection successful!")
    print(f"Result: {result}")
    
    # Test with parameters
    param_result = db_manager.execute_query(
        "SELECT :value as param_test;", 
        {"value": 42}
    )
    print(f"Parameter test result: {param_result}")
    
except Exception as e:
    print(f"Connection failed: {str(e)}")
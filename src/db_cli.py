#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Database CLI Tool for Factor Model Framework

This script provides command line utilities for managing the factor model database.
"""

import os
import sys
import logging
import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import dotenv

# Import database components
from db_manager import DatabaseManager, DatabaseConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def init_db_connection():
    """Initialize database connection"""
    # Load environment variables
    dotenv.load_dotenv()
    
    db_config = DatabaseConfig(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        username=os.getenv("DB_USERNAME"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        schema=os.getenv("DB_SCHEMA", "public")
    )
    
    db_manager = DatabaseManager(db_config)
    return db_manager

def init_db(args):
    """Initialize the database schema"""
    db_manager = init_db_connection()
    
    script_path = args.script_path
    if not script_path:
        # Use default schema file in current directory
        script_path = os.path.join(os.path.dirname(__file__), "db_schema.sql")
    
    if not os.path.exists(script_path):
        logger.error(f"Schema file not found: {script_path}")
        return 1
    
    try:
        logger.info(f"Initializing database with schema from {script_path}")
        db_manager.execute_script(script_path)
        logger.info("Database schema initialized successfully")
        return 0
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}", exc_info=True)
        return 1

def export_data(args):
    """Export data from the database to CSV files"""
    db_manager = init_db_connection()
    
    export_dir = args.output_dir
    if not export_dir:
        export_dir = os.path.join(os.getcwd(), "exports", datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    # Create export directory if it doesn't exist
    os.makedirs(export_dir, exist_ok=True)
    
    tables = args.tables.split(",") if args.tables else [
        "securities",
        "universes",
        "universe_constituents",
        "factors",
        "factor_data",
        "security_prices",
        "security_returns",
        "backtest_configs",
        "portfolio_weights",
        "backtest_results"
    ]
    
    success = True
    for table in tables:
        try:
            logger.info(f"Exporting table: {table}")
            query = f"SELECT * FROM {table};"
            df = db_manager.get_dataframe(query)
            
            if df.empty:
                logger.warning(f"No data found in table: {table}")
                continue
            
            output_file = os.path.join(export_dir, f"{table}.csv")
            df.to_csv(output_file, index=False)
            logger.info(f"Exported {len(df)} rows to {output_file}")
        except Exception as e:
            logger.error(f"Error exporting table {table}: {str(e)}")
            success = False
    
    if success:
        logger.info(f"Data export completed successfully to {export_dir}")
        return 0
    else:
        logger.error("Data export completed with errors")
        return 1

def import_data(args):
    """Import data from CSV files into the database"""
    db_manager = init_db_connection()
    
    import_dir = args.input_dir
    if not os.path.exists(import_dir):
        logger.error(f"Import directory not found: {import_dir}")
        return 1
    
    tables = args.tables.split(",") if args.tables else None
    
    # Get all CSV files in the import directory
    csv_files = [f for f in os.listdir(import_dir) if f.endswith('.csv')]
    
    if not csv_files:
        logger.error(f"No CSV files found in {import_dir}")
        return 1
    
    success = True
    for csv_file in csv_files:
        table_name = os.path.splitext(csv_file)[0]
        
        # Skip if not in the specified tables
        if tables and table_name not in tables:
            continue
        
        try:
            logger.info(f"Importing data into table: {table_name}")
            file_path = os.path.join(import_dir, csv_file)
            df = pd.read_csv(file_path)
            
            if df.empty:
                logger.warning(f"No data found in file: {csv_file}")
                continue
            
            # Convert date columns to datetime
            for col in df.columns:
                if col.lower().endswith('date'):
                    df[col] = pd.to_datetime(df[col])
            
            # Import data
            db_manager.insert_dataframe(df, table_name, if_exists='append')
            logger.info(f"Imported {len(df)} rows into {table_name}")
        except Exception as e:
            logger.error(f"Error importing data into {table_name}: {str(e)}")
            success = False
    
    if success:
        logger.info("Data import completed successfully")
        return 0
    else:
        logger.error("Data import completed with errors")
        return 1

def backup_db(args):
    """Create a full database backup"""
    db_manager = init_db_connection()
    
    backup_dir = args.output_dir
    if not backup_dir:
        backup_dir = os.path.join(os.getcwd(), "backups", datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    # Create backup directory if it doesn't exist
    os.makedirs(backup_dir, exist_ok=True)
    
    # Get list of all tables
    query = """
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'public' 
    AND table_type = 'BASE TABLE';
    """
    tables_df = db_manager.get_dataframe(query)
    
    if tables_df.empty:
        logger.error("No tables found in database")
        return 1
    
    success = True
    for _, row in tables_df.iterrows():
        table = row['table_name']
        try:
            logger.info(f"Backing up table: {table}")
            query = f"SELECT * FROM {table};"
            df = db_manager.get_dataframe(query)
            
            output_file = os.path.join(backup_dir, f"{table}.csv")
            df.to_csv(output_file, index=False)
            logger.info(f"Backed up {len(df)} rows from {table}")
        except Exception as e:
            logger.error(f"Error backing up table {table}: {str(e)}")
            success = False
    
    # Create metadata file
    metadata = {
        "backup_date": datetime.now().isoformat(),
        "tables": tables_df['table_name'].tolist(),
        "host": db_manager.config.host,
        "database": db_manager.config.database,
        "schema": db_manager.config.schema
    }
    
    with open(os.path.join(backup_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    if success:
        logger.info(f"Database backup completed successfully to {backup_dir}")
        return 0
    else:
        logger.error("Database backup completed with errors")
        return 1

def restore_db(args):
    """Restore database from a backup"""
    db_manager = init_db_connection()
    
    backup_dir = args.input_dir
    if not os.path.exists(backup_dir):
        logger.error(f"Backup directory not found: {backup_dir}")
        return 1
    
    # Check for metadata file
    metadata_file = os.path.join(backup_dir, "metadata.json")
    if not os.path.exists(metadata_file):
        logger.warning("Metadata file not found, proceeding without verification")
    else:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            logger.info(f"Restoring backup from {metadata['backup_date']}")
    
    # Get all CSV files in the backup directory
    csv_files = [f for f in os.listdir(backup_dir) if f.endswith('.csv')]
    
    if not csv_files:
        logger.error(f"No CSV files found in {backup_dir}")
        return 1
    
    if args.clear_tables:
        logger.warning("Clearing existing data from tables before restore")
        for csv_file in csv_files:
            table_name = os.path.splitext(csv_file)[0]
            try:
                query = f"TRUNCATE TABLE {table_name} CASCADE;"
                db_manager.execute_query(query)
                logger.info(f"Cleared data from table: {table_name}")
            except Exception as e:
                logger.error(f"Error clearing table {table_name}: {str(e)}")
    
    success = True
    for csv_file in csv_files:
        table_name = os.path.splitext(csv_file)[0]
        try:
            logger.info(f"Restoring data into table: {table_name}")
            file_path = os.path.join(backup_dir, csv_file)
            df = pd.read_csv(file_path)
            
            if df.empty:
                logger.warning(f"No data found in file: {csv_file}")
                continue
            
            # Convert date columns to datetime
            for col in df.columns:
                if col.lower().endswith('date'):
                    df[col] = pd.to_datetime(df[col])
            
            # Restore data
            if args.clear_tables:
                db_manager.insert_dataframe(df, table_name, if_exists='append')
            else:
                # Use a temporary table and then insert with conflict handling
                temp_table = f"temp_{table_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                db_manager.insert_dataframe(df, temp_table, if_exists='replace')
                
                # Get primary key columns
                query = f"""
                SELECT a.attname
                FROM pg_index i
                JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
                WHERE i.indrelid = '{table_name}'::regclass
                AND i.indisprimary;
                """
                pk_cols = db_manager.get_dataframe(query)
                
                if not pk_cols.empty:
                    pk_list = ", ".join(pk_cols['attname'])
                    
                    # Insert with conflict handling
                    query = f"""
                    INSERT INTO {table_name}
                    SELECT * FROM {temp_table}
                    ON CONFLICT ({pk_list}) DO UPDATE
                    SET {", ".join([f"{col} = EXCLUDED.{col}" for col in df.columns if col not in pk_cols['attname'].tolist()])}
                    """
                    db_manager.execute_query(query)
                else:
                    # No primary key, just insert
                    db_manager.insert_dataframe(df, table_name, if_exists='append')
                
                # Drop temporary table
                db_manager.execute_query(f"DROP TABLE IF EXISTS {temp_table};")
            
            logger.info(f"Restored {len(df)} rows into {table_name}")
        except Exception as e:
            logger.error(f"Error restoring data into {table_name}: {str(e)}")
            success = False
    
    if success:
        logger.info("Database restore completed successfully")
        return 0
    else:
        logger.error("Database restore completed with errors")
        return 1

def clean_db(args):
    """Clean old data from the database"""
    db_manager = init_db_connection()
    
    retention_days = args.retention_days
    tables = args.tables.split(",") if args.tables else [
        "factor_data",
        "security_prices",
        "security_returns",
        "portfolio_weights",
        "backtest_results"
    ]
    
    cutoff_date = (datetime.now() - timedelta(days=retention_days)).strftime("%Y-%m-%d")
    
    success = True
    for table in tables:
        try:
            # Check if table has a date column
            query = f"""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = '{table}'
            AND column_name = 'date';
            """
            date_col = db_manager.get_dataframe(query)
            
            if date_col.empty:
                logger.warning(f"Table {table} does not have a date column, skipping")
                continue
            
            # Count rows before deletion
            count_query = f"SELECT COUNT(*) as count FROM {table} WHERE date < '{cutoff_date}';"
            count_df = db_manager.get_dataframe(count_query)
            count = count_df['count'].iloc[0] if not count_df.empty else 0
            
            if count == 0:
                logger.info(f"No data to clean in table {table}")
                continue
            
            # Delete old data
            delete_query = f"DELETE FROM {table} WHERE date < '{cutoff_date}';"
            db_manager.execute_query(delete_query)
            
            logger.info(f"Cleaned {count} rows from {table} older than {cutoff_date}")
        except Exception as e:
            logger.error(f"Error cleaning table {table}: {str(e)}")
            success = False
    
    if success:
        logger.info("Database cleaning completed successfully")
        return 0
    else:
        logger.error("Database cleaning completed with errors")
        return 1

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Database CLI Tool for Factor Model Framework")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # init command
    init_parser = subparsers.add_parser("init", help="Initialize database schema")
    init_parser.add_argument("--script-path", type=str, help="Path to SQL schema script")
    
    # export command
    export_parser = subparsers.add_parser("export", help="Export data to CSV files")
    export_parser.add_argument("--output-dir", type=str, help="Output directory for exported files")
    export_parser.add_argument("--tables", type=str, help="Comma-separated list of tables to export")
    
    # import command
    import_parser = subparsers.add_parser("import", help="Import data from CSV files")
    import_parser.add_argument("input_dir", type=str, help="Input directory with CSV files")
    import_parser.add_argument("--tables", type=str, help="Comma-separated list of tables to import")
    
    # backup command
    backup_parser = subparsers.add_parser("backup", help="Create a full database backup")
    backup_parser.add_argument("--output-dir", type=str, help="Output directory for backup files")
    
    # restore command
    restore_parser = subparsers.add_parser("restore", help="Restore database from backup")
    restore_parser.add_argument("input_dir", type=str, help="Input directory with backup files")
    restore_parser.add_argument("--clear-tables", action="store_true", help="Clear tables before restore")
    
    # clean command
    clean_parser = subparsers.add_parser("clean", help="Clean old data from database")
    clean_parser.add_argument("--retention-days", type=int, default=365, help="Days to retain data (default: 365)")
    clean_parser.add_argument("--tables", type=str, help="Comma-separated list of tables to clean")
    
    args = parser.parse_args()
    
    if args.command == "init":
        return init_db(args)
    elif args.command == "export":
        return export_data(args)
    elif args.command == "import":
        return import_data(args)
    elif args.command == "backup":
        return backup_db(args)
    elif args.command == "restore":
        return restore_db(args)
    elif args.command == "clean":
        return clean_db(args)
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())
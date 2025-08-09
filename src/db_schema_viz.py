#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Database Schema Visualization

This script generates an entity-relationship diagram for the factor model database
using SQLAlchemy and graphviz.
"""

import os
import sys
import logging
import argparse
import dotenv
from sqlalchemy import create_engine, MetaData, inspect, Table, Column, ForeignKey
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
import graphviz

# Import database configuration
from db_manager import DatabaseConfig

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
    
    # Create engine
    conn_str = db_config.get_connection_string()
    engine = create_engine(conn_str)
    
    return engine, db_config.schema

def generate_er_diagram(engine, schema, output_file=None, format="png", show_columns=True, show_datatypes=True):
    """
    Generate entity-relationship diagram
    
    Args:
        engine: SQLAlchemy engine
        schema: Database schema
        output_file: Output file path
        format: Output format (png, pdf, svg)
        show_columns: Whether to show columns
        show_datatypes: Whether to show datatypes
    """
    metadata = MetaData(schema=schema)
    metadata.reflect(bind=engine)
    
    # Create graphviz object
    graph = graphviz.Digraph(
        name="Factor_Model_Database",
        comment="Entity-Relationship Diagram for Factor Model Database",
        format=format,
        engine="dot"
    )
    graph.attr(rankdir="LR", size="8,5", ratio="fill", concentrate="true")
    graph.attr("node", shape="plain", style="filled", fillcolor="lightblue")
    graph.attr("edge", color="darkblue", style="dashed", arrowhead="normal", arrowsize="0.5")
    
    # Add tables
    for table_name, table in metadata.tables.items():
        table_name = table_name.split(".")[-1]  # Remove schema prefix
        
        # Create HTML-like label for table
        label = f'<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">'
        label += f'<TR><TD COLSPAN="3" BGCOLOR="#4472C4"><FONT COLOR="white"><B>{table_name}</B></FONT></TD></TR>'
        
        if show_columns:
            label += '<TR><TD BGCOLOR="#D6DFEC"><B>Column</B></TD>'
            if show_datatypes:
                label += '<TD BGCOLOR="#D6DFEC"><B>Type</B></TD>'
            label += '<TD BGCOLOR="#D6DFEC"><B>PK/FK</B></TD></TR>'
            
            # Get primary key columns
            inspector = inspect(engine)
            pk_columns = [pk['name'] for pk in inspector.get_pk_constraint(table_name, schema=schema).get('constrained_columns', [])]
            
            # Add columns
            for column in table.columns:
                col_name = column.name
                col_type = str(column.type)
                col_pk = "PK" if col_name in pk_columns else ""
                col_fk = "FK" if isinstance(column.foreign_keys, set) and len(column.foreign_keys) > 0 else ""
                col_constraints = f"{col_pk}{' ' if col_pk and col_fk else ''}{col_fk}"
                
                label += f'<TR><TD>{col_name}</TD>'
                if show_datatypes:
                    label += f'<TD>{col_type}</TD>'
                label += f'<TD>{col_constraints}</TD></TR>'
        
        label += '</TABLE>>'
        
        graph.node(table_name, label=label)
    
    # Add relationships
    for table_name, table in metadata.tables.items():
        table_name = table_name.split(".")[-1]  # Remove schema prefix
        
        for fk in table.foreign_key_constraints:
            ref_table = fk.referred_table.name.split(".")[-1]  # Remove schema prefix
            
            graph.edge(
                ref_table,
                table_name,
                label="",
                headlabel=" ",
                taillabel=" "
            )
    
    # Render graph
    if output_file:
        graph.render(output_file, cleanup=True, view=False)
        logger.info(f"Generated ER diagram: {output_file}.{format}")
        return f"{output_file}.{format}"
    else:
        output_file = "factor_model_db_schema"
        graph.render(output_file, cleanup=True, view=False)
        logger.info(f"Generated ER diagram: {output_file}.{format}")
        return f"{output_file}.{format}"

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate ER diagram for Factor Model Database")
    parser.add_argument("--output", "-o", type=str, help="Output file path (without extension)")
    parser.add_argument("--format", "-f", type=str, default="png", choices=["png", "pdf", "svg"], help="Output format")
    parser.add_argument("--no-columns", action="store_true", help="Don't show columns")
    parser.add_argument("--no-datatypes", action="store_true", help="Don't show datatypes")
    
    args = parser.parse_args()
    
    try:
        engine, schema = init_db_connection()
        
        output_file = generate_er_diagram(
            engine=engine,
            schema=schema,
            output_file=args.output,
            format=args.format,
            show_columns=not args.no_columns,
            show_datatypes=not args.no_datatypes
        )
        
        logger.info(f"ER diagram generated successfully: {output_file}")
        return 0
    except Exception as e:
        logger.error(f"Error generating ER diagram: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
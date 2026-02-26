import uuid
from cassandra.cqlengine import columns
from cassandra.cqlengine import connection
from datetime import datetime
from cassandra.cqlengine.management import sync_table
from cassandra.cqlengine.models import Model
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster
from utils.logger import ph3_1_logger
import pytz

from utils.config_reader import ConfigReader

cfg = ConfigReader()
cass = cfg.get("cassandra_props") or cfg.get("cassandra") or {}

cassandra_host = cass.get("host")
username = cass.get("username")
password = cass.get("password")
keyspace = cass.get("keyspace")

auth_provider = PlainTextAuthProvider(username=username, password=password) if username and password else None

connection.setup(
    hosts=[cassandra_host],
    default_keyspace=keyspace,
    auth_provider=auth_provider,
    protocol_version=4,
    lazy_connect=True,
    retry_connect=True,
    connect_timeout=20,
)

cluster = Cluster([cassandra_host], auth_provider=auth_provider)
session = cluster.connect(keyspace)


# Check if keyspace exists, create if it doesn't
try:
    session.execute(f"SELECT keyspace_name FROM system_schema.keyspaces WHERE keyspace_name = '{keyspace}'")
    ph3_1_logger.info(f"Keyspace {keyspace} exists")
except Exception as e:
    ph3_1_logger.warning(f"Keyspace {keyspace} does not exist, creating it")
    try:
        session.execute(f"""
            CREATE KEYSPACE {keyspace}
            WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': 1}}
        """)
        ph3_1_logger.info(f"Keyspace {keyspace} created successfully")
    except Exception as e:
        ph3_1_logger.error(f"Failed to create keyspace {keyspace}: {str(e)}", exc_info=True)
        raise

def to_datetime(ms, source_tz=pytz.UTC):
    """
    Convert a timestamp in milliseconds to a UTC-aware datetime object.
    
    Args:
        ms: Timestamp in milliseconds (int, float, or None)
        source_tz: Source timezone (default: UTC)
    
    Returns:
        datetime: UTC-aware datetime object, or None if ms is None
    """
    return datetime.fromtimestamp(ms / 1000, tz=source_tz).astimezone(pytz.UTC) if ms else None

# Schema definitions
class ProductionCorrelation(Model):
    __table_name__ = "dw_tbl_production_correlation"
    workstation_id = columns.Integer(partition_key=True)
    output_stock_id = columns.Text(partition_key=True)
    process_id = columns.Text(primary_key=True, clustering_order="DESC")
    process_timestamp = columns.DateTime(primary_key=True, clustering_order="DESC")
    quantity = columns.Double(required=True)

    @classmethod
    def saveData(cls, *, message):
        try:
            workstation_id = int(message.get('workStationId', 0))
            if not workstation_id:
                ph3_1_logger.error("Missing workStationId in message")
                return None

            produce_list = message.get('produceList', [])
            if not produce_list:
                ph3_1_logger.error("Empty produceList in message")
                return None

            # Use the first product in produce_list
            product = produce_list[0]
            output_stock_id = str(product.get('stId', ''))
            if not output_stock_id:
                ph3_1_logger.error("Missing stId in produceList")
                return None

            # Use pre-generated process_id
            process_id = message.get('joOpId', '')
            if not process_id:
                ph3_1_logger.error("Missing process_id in message")
                return None

            # Handle timestamp
            try:
                create_date = message.get('createDate', None)
                if isinstance(create_date, (int, float)):
                    timestamp = to_datetime(create_date, source_tz=pytz.timezone('Asia/Karachi'))  # Assume PKT input
                    if timestamp is None:
                        ph3_1_logger.error("Invalid createDate: None returned from to_datetime")
                        timestamp = datetime.now(pytz.UTC)
                elif isinstance(create_date, str):
                    # Parse ISO format string and ensure UTC
                    timestamp = datetime.fromisoformat(create_date.replace('Z', '+00:00')).astimezone(pytz.UTC)
                elif isinstance(create_date, datetime):
                    # Ensure timezone awareness
                    timestamp = create_date.astimezone(pytz.UTC) if create_date.tzinfo else create_date.replace(tzinfo=pytz.UTC)
                else:
                    ph3_1_logger.error(f"Invalid createDate type: {type(create_date)}, using current time")
                    timestamp = datetime.now(pytz.UTC)
            except (TypeError, ValueError) as e:
                ph3_1_logger.error(f"Invalid createDate: {message.get('createDate')}, using current time")
                timestamp = datetime.now(pytz.UTC)

            # Parse quantity
            try:
                quantity = float(product.get('qty', 0))
            except (ValueError, TypeError) as e:
                ph3_1_logger.error(f"Invalid quantity: {str(e)}")
                return None

            # Create record
            record = cls.create(
                workstation_id=workstation_id,
                output_stock_id=output_stock_id,
                process_id=process_id,
                process_timestamp=timestamp,
                quantity=quantity
            )
            ph3_1_logger.debug(f"Saved ProductionCorrelation: {record.process_id} with timestamp {timestamp}")
            return record

        except Exception as e:
            ph3_1_logger.error(f"saveData failed: {str(e)}", exc_info=True)
            return None

class ProductionCorrelationSummary(Model):
    __table_name__ = "dw_tbl_production_correlation_summary"
    workstation_id = columns.Integer(primary_key=True)
    output_stock_id = columns.Text(primary_key=True)
    total_quantity = columns.Double(required=True)
    process_count = columns.Integer(required=True)
    average_quantity = columns.Double(required=True)
    first_production = columns.DateTime(required=True)
    last_update = columns.DateTime(required=True)

    @classmethod
    def update_summary(cls, process_record):
        try:
            existing = cls.objects(
                workstation_id=process_record.workstation_id,
                output_stock_id=process_record.output_stock_id
            ).first()

            quantity = process_record.quantity
            timestamp = process_record.process_timestamp

            # Validate input
            if not isinstance(timestamp, datetime):
                ph3_1_logger.error(f"Invalid timestamp type: {type(timestamp)}, using current time")
                timestamp = datetime.now(pytz.UTC)
            elif timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=pytz.UTC)  # Silently assume UTC

            if not isinstance(quantity, (int, float)):
                ph3_1_logger.error(f"Invalid quantity type: {quantity} (must be numeric)")
                return None
            if quantity < 0:
                ph3_1_logger.error(f"Negative quantity: {quantity} (not allowed)")
                return None

            # Validate timestamp to prevent invalid dates (e.g., before 1970)
            min_valid_date = datetime(1970, 1, 1, tzinfo=pytz.UTC)
            if timestamp < min_valid_date:
                ph3_1_logger.error(f"Invalid timestamp {timestamp}, using current time")
                timestamp = datetime.now(pytz.UTC)

            if existing:
                # Ensure existing timestamps are timezone aware, assuming UTC
                existing_first = existing.first_production
                existing_last = existing.last_update
                if existing_first.tzinfo is None:
                    existing_first = existing_first.replace(tzinfo=pytz.UTC)  # Silently convert
                if existing_last.tzinfo is None:
                    existing_last = existing_last.replace(tzinfo=pytz.UTC)  # Silently convert

                # Update summary values
                total_quantity = existing.total_quantity + quantity
                process_count = existing.process_count + 1
                average_quantity = total_quantity / process_count

                # Set first_production to the earliest valid timestamp
                first_production = min(existing_first, timestamp)
                last_update = max(existing_last, timestamp)

                existing.update(
                    total_quantity=total_quantity,
                    process_count=process_count,
                    average_quantity=average_quantity,
                    first_production=first_production,
                    last_update=last_update
                )
                ph3_1_logger.debug(f"Updated ProductionCorrelationSummary: {existing}")
                return existing
            else:
                # Create new summary
                new_summary = cls.create(
                    workstation_id=process_record.workstation_id,
                    output_stock_id=process_record.output_stock_id,
                    total_quantity=quantity,
                    process_count=1,
                    average_quantity=quantity,
                    first_production=timestamp,
                    last_update=timestamp
                )
                ph3_1_logger.debug(f"Created ProductionCorrelationSummary: {new_summary}")
                return new_summary

        except Exception as e:
            ph3_1_logger.error(f"update_summary failed: {str(e)}", exc_info=True)
            return None

# Sync tables to create them if they don't exist
sync_table(ProductionCorrelation)
ph3_1_logger.info("ProductionCorrelation table synced/created")
sync_table(ProductionCorrelationSummary)
ph3_1_logger.info("ProductionCorrelationSummary table synced/created")
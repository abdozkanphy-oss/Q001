
from cassandra.cqlengine import columns
from cassandra.cqlengine.models import Model
from cassandra.cqlengine import connection
from cassandra.cqlengine.query import SimpleStatement
from cassandra.auth import PlainTextAuthProvider
from utils.logger import ph1_logger
import json
from datetime import datetime
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


class AnomalyModel(Model):
    __keyspace__ = keyspace
    __table_name__ = None
    month_year = columns.Text(primary_key=True)
    start_time = columns.DateTime(primary_key=True, clustering_order="DESC")
    shift_start_text = columns.Text(primary_key=True, clustering_order="DESC")
    message_key = columns.Text(primary_key=True, clustering_order="ASC")
    anomaly_importance_avg = columns.Double()
    anomaly_importance_max = columns.Double()
    anomaly_importance_min = columns.Double()
    count_high_anomalies = columns.Integer()
    count_low_anomalies = columns.Integer()
    count_normal_values = columns.Integer()
    count_total_anomalies = columns.Integer()
    shift_start_time = columns.DateTime()
    shift_finish_text = columns.Text()
    shift_finish_time = columns.DateTime()
    customer = columns.Text()
    data_trend_avg = columns.Double()
    data_trend_max = columns.Double()
    data_trend_min = columns.Double()
    equipment_id = columns.Integer()
    equipment_measuring_point_id = columns.Integer()
    equipment_name = columns.Text()
    equipment_no = columns.Text()
    equipment_predicted_lower_limit_avg = columns.Double()
    equipment_predicted_lower_limit_max = columns.Double()
    equipment_predicted_lower_limit_min = columns.Double()
    equipment_predicted_upper_limit_avg = columns.Double()
    equipment_predicted_upper_limit_max = columns.Double()
    equipment_predicted_upper_limit_min = columns.Double()
    equipment_predicted_value_avg = columns.Double()
    equipment_predicted_value_std_dev = columns.Double()
    equipment_real_value_avg = columns.Double()
    equipment_real_value_max = columns.Double()
    equipment_real_value_min = columns.Double()
    equipment_real_value_std_dev = columns.Double()
    finish_time = columns.DateTime()
    job_order_operation_id = columns.Integer()
    machine_state = columns.Text()
    measurement_count = columns.BigInt()
    organization_id = columns.Integer()
    organization_name = columns.Text()
    organization_no = columns.Text()
    plant_id = columns.Integer()
    plant_name = columns.Text()
    produced_stock_id = columns.Integer()
    produced_stock_name = columns.Text()
    produced_stock_no = columns.Text()
    work_center_id = columns.Integer()
    work_center_name = columns.Text()
    work_center_no = columns.Text()
    work_station_id = columns.Integer()
    work_station_name = columns.Text()
    work_station_no = columns.Text()
    workstation_state = columns.Text()
    year = columns.Text()

    @classmethod
    def saveData(cls, message, main_message, y, key, table_name):
        ph1_logger.debug(f"saveData - Setting table_name to {table_name}")
        cls.__table_name__ = table_name

        # Define valid columns for each table
        table_columns = {
            "tbl_shiftly_anomaly": {
                "primary_keys": ["month_year", "shift_start_text", "message_key"],
                "exclude": ["start_time", "finish_time"]
            },
            "tbl_one_minute_anomaly": {
                "primary_keys": ["month_year", "start_time", "message_key"],
                "exclude": ["shift_start_text", "shift_start_time", "shift_finish_text", "shift_finish_time"]
            },
            "tbl_ten_minute_anomaly": {
                "primary_keys": ["month_year", "start_time", "message_key"],
                "exclude": ["shift_start_text", "shift_start_time", "shift_finish_text", "shift_finish_time"]
            },
            "tbl_one_day_anomaly": {
                "primary_keys": ["month_year", "start_time", "message_key"],
                "exclude": ["shift_start_text", "shift_start_time", "shift_finish_text", "shift_finish_time"]
            },
            "tbl_one_hour_anomaly": {
                "primary_keys": ["month_year", "start_time", "message_key"],
                "exclude": ["shift_start_text", "shift_start_time", "shift_finish_text", "shift_finish_time"]
            },
            "tbl_one_week_anomaly": {
                "primary_keys": ["month_year", "start_time", "message_key"],
                "exclude": ["shift_start_text", "shift_start_time", "shift_finish_text", "shift_finish_time"]
            }
        }

        # Validate table_name
        if table_name not in table_columns:
            ph1_logger.error(f"saveData - Invalid table_name: {table_name}")
            raise ValueError(f"Invalid table_name: {table_name}")

        valid_columns = table_columns[table_name]
        ph1_logger.debug(f"saveData - Valid columns for {table_name}: {valid_columns}")

        # Ensure shift fields for tbl_shiftly_anomaly
        if table_name == "tbl_shiftly_anomaly" and not message.get("shift_start_time"):
            ph1_logger.warning(f"saveData - Missing shift_start_time for tbl_shiftly_anomaly, using measDt")
            message["shift_start_time"] = message.get("measDt")

        if y == 0:
            upper = 1.0
            lower = -1.0
        else:
            upper = y + y * 0.05
            lower = y - y * 0.05
        reading = float(message.get("cntRead", 0)) if message.get("cntRead") else 0
        anomaly = 1 if reading > upper or reading < lower else 0

        meas_dt = message.get("measDt")
        start_time = datetime.fromtimestamp(int(meas_dt) / 1000, tz=pytz.UTC) if meas_dt and meas_dt.isdigit() else datetime.now(pytz.UTC)
        month_year = f"{start_time.month}-{start_time.year}"
        year = message.get("year", start_time.strftime("%Y"))

        def get_main_attr(attr, default=None):
            try:
                value = getattr(main_message, attr, None)
                if value is None:
                    camel_case = ''.join(word.capitalize() for word in attr.split('_'))
                    camel_case = camel_case[0].lower() + camel_case[1:]
                    value = getattr(main_message, camel_case, None)
                if value is not None:
                    return value
            except AttributeError:
                pass
            if isinstance(main_message, dict):
                value = main_message.get(attr, None)
                if value is None:
                    camel_case = ''.join(word.capitalize() for word in attr.split('_'))
                    camel_case = camel_case[0].lower() + camel_case[1:]
                    value = main_message.get(camel_case, None)
                if value is not None:
                    return value
            return default

        values = {
            "month_year": month_year,
            "start_time": start_time if "start_time" in valid_columns["primary_keys"] else None,
            "shift_start_text": None,
            "message_key": key,
            "anomaly_importance_avg": float(anomaly),
            "anomaly_importance_max": float(anomaly),
            "anomaly_importance_min": float(anomaly),
            "count_high_anomalies": 1 if reading > upper else 0,
            "count_low_anomalies": 1 if reading < lower else 0,
            "count_normal_values": 1 if not anomaly else 0,
            "count_total_anomalies": int(anomaly),
            "customer": str(message.get("cust", "turna")),
            "data_trend_avg": float(y),
            "data_trend_max": float(y),
            "data_trend_min": float(y),
            "equipment_id": int(message.get("eqId", 0)) if message.get("eqId") else 0,
            "equipment_measuring_point_id": int(message.get("eqMeasPtId", 0)) if message.get("eqMeasPtId") else 0,
            "equipment_name": str(message.get("eqNm", "")),
            "equipment_no": str(message.get("eqNo", "")),
            "equipment_predicted_lower_limit_avg": float(lower),
            "equipment_predicted_lower_limit_max": float(lower),
            "equipment_predicted_lower_limit_min": float(lower),
            "equipment_predicted_upper_limit_avg": float(upper),
            "equipment_predicted_upper_limit_max": float(upper),
            "equipment_predicted_upper_limit_min": float(upper),
            "equipment_predicted_value_avg": float(y),
            "equipment_predicted_value_std_dev": float(0.05 * y) if y != 0 else 0.1,
            "equipment_real_value_avg": reading,
            "equipment_real_value_max": reading,
            "equipment_real_value_min": reading,
            "equipment_real_value_std_dev": float(0.0),
            "finish_time": get_main_attr("create_date", datetime.now(pytz.UTC)) if table_name != "tbl_shiftly_anomaly" else None,
            "job_order_operation_id": int(message.get("joOpId", 0)),
            "machine_state": str(get_main_attr("machine_state", "")),
            "measurement_count": int(message.get("measurement_count", 1)),
            "organization_id": int(message.get("orgId", 0)),
            "organization_name": str(message.get("orgNm", "")),
            "organization_no": str(message.get("orgNo", "")),
            "plant_id": int(message.get("plId", 0)),
            "plant_name": str(message.get("plNm", "")),
            "produced_stock_id": int(get_main_attr("produced_stock_id", 0)) if get_main_attr("produced_stock_id") is not None else 0,
            "produced_stock_name": str(get_main_attr("produced_stock_name", "")),
            "produced_stock_no": str(get_main_attr("produced_stock_no", "")),
            "work_center_id": int(get_main_attr("work_center_id", 0)) if get_main_attr("work_center_id") is not None else 0,
            "work_center_name": str(message.get("wcNm", "")),
            "work_center_no": str(get_main_attr("work_center_no", "")),
            "work_station_id": int(get_main_attr("work_station_id", 0)) if get_main_attr("work_station_id") is not None else 0,
            "work_station_name": str(get_main_attr("work_station_name", "")),
            "work_station_no": str(get_main_attr("work_station_no", "")),
            "workstation_state": str(get_main_attr("workstation_state", "")),
            "year": year,
        }

        if table_name == "tbl_shiftly_anomaly":
            shift_start_time = get_main_attr("shift_start_time", start_time)
            shift_finish_time = get_main_attr("shift_finish_time", start_time)
            values.update({
                "shift_start_text": str(int(shift_start_time.timestamp() * 1000)),
                "shift_start_time": shift_start_time,
                "shift_finish_text": str(int(shift_finish_time.timestamp() * 1000)),
                "shift_finish_time": shift_finish_time,
            })

        # Filter values to include only valid columns
        clean_values = {k: v for k, v in values.items() if v is not None and k not in valid_columns.get("exclude", [])}
        # ph1_logger.debug(f"saveData - Saving to {table_name} with values: {json.dumps({k: str(v) for k, v in clean_values.items()}, indent=2)}")

        # Ensure correct table
        if table_name != cls.__table_name__:
            ph1_logger.error(f"saveData - Table mismatch before query: requested {table_name}, but model set to {cls.__table_name__}")
            cls.__table_name__ = table_name

        try:
            # Generate and log the CQL query
            columns = ', '.join(clean_values.keys())
            placeholders = ', '.join(['%s'] * len(clean_values))
            query = f"INSERT INTO {keyspace}.{table_name} ({columns}) VALUES ({placeholders})"
            # ph1_logger.debug(f"saveData - Executing CQL query: {query} with values: {list(clean_values.values())}")
            
            # Use SimpleStatement to ensure correct table
            statement = SimpleStatement(query)
            session = connection.get_session()
            result = session.execute(statement, list(clean_values.values()))
            
            ph1_logger.debug(f"saveData - Successfully saved to {table_name}")
            return result
        except Exception as e:
            ph1_logger.error(f"saveData - Failed to save to {table_name}: {e}")
            raise

def save_anomaly_data(message, main_message, y, key, granularity="one_day"):
    ph1_logger.debug(f"add_output_predict - Input to saveData: message={message}, main_message={main_message}, y={y}, key={key}")
    table_map = {
        "one_day": "tbl_one_day_anomaly",
        "one_hour": "tbl_one_hour_anomaly",
        "one_minute": "tbl_one_minute_anomaly",
        "one_week": "tbl_one_week_anomaly",
        "shiftly": "tbl_shiftly_anomaly",
        "ten_minute": "tbl_ten_minute_anomaly"
    }
    
    if granularity not in table_map:
        ph1_logger.error(f"add_output_predict - Invalid granularity: {granularity}")
        raise ValueError(f"Invalid granularity: {granularity}. Must be one of {list(table_map.keys())}")

    table_name = table_map[granularity]
    ph1_logger.debug(f"add_output_predict - Mapping granularity {granularity} to table {table_name}")

    try:
        data = AnomalyModel.saveData(
            message=message,
            main_message=main_message,
            y=y,
            key=key,
            table_name=table_name
        )
        ph1_logger.debug(f"add_output_predict - The data is saved in table {table_name}")
        return data
    except Exception as e:
        ph1_logger.error(f"add_output_predict - Failed to save to {table_name}: {e}")
        raise




import uuid
from cassandra.cqlengine import columns
from cassandra.cqlengine.models import Model
from cassandra.cqlengine import connection
from cassandra.cqlengine.query import SimpleStatement
from cassandra.auth import PlainTextAuthProvider
from utils.logger import ph2_logger
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

class phaseTwoReport(Model):
    __keyspace__ = keyspace
    __table_name__ = None
    year_month = columns.Text(primary_key=True)
    partition_date = columns.Text(primary_key=True, clustering_order="DESC")
    measurement_date = columns.DateTime(primary_key=True, clustering_order="DESC")
    unique_code = columns.Text(primary_key=True, clustering_order="DESC")
    active = columns.Boolean()
    anomaly = columns.Integer()
    anomaly_detection_active = columns.Boolean()
    anomaly_importance = columns.Double()
    anomaly_score = columns.Double()
    bias = columns.Double()
    coefficient = columns.Double()
    counter_reading = columns.Double()
    create_auto_maintenance_order_notification = columns.Boolean()
    create_date = columns.DateTime()
    customer = columns.Text()
    detected_anomaly = columns.Double()
    employee_id = columns.Integer()
    equipment_code_group_header_id = columns.Integer()
    equipment_id = columns.Integer()
    equipment_measuring_point_id = columns.Integer()
    equipment_name = columns.Text()
    equipment_no = columns.Text()
    id = columns.UUID()
    job_order_operation_id = columns.Integer()
    machine_state = columns.Text()
    max_error_count_in_hour = columns.Integer()
    max_error_count_in_min = columns.Integer()
    max_error_count_in_shift = columns.Integer()
    mean_square_error = columns.Double()
    measurement_document_id = columns.Integer()
    message_key = columns.Text()
    month_year = columns.Text()
    organization_id = columns.Integer()
    organization_name = columns.Text()
    organization_no = columns.Text()
    parameter = columns.Text()
    plant_id = columns.Integer()
    plant_name = columns.Text()
    plant_no = columns.Text()
    predicted_lower = columns.Double()
    predicted_upper = columns.Double()
    predicted_value = columns.Double()
    produced_stock_id = columns.Integer()
    produced_stock_name = columns.Text()
    produced_stock_no = columns.Text()
    regression_value = columns.Double()
    shift_finish_text = columns.Text()
    shift_finish_time = columns.DateTime()
    shift_start_text = columns.Text()
    shift_start_time = columns.DateTime()
    stock_id = columns.Integer()
    trend = columns.Double()
    trend_calculation_active = columns.Boolean()
    trend_lower = columns.Double()
    trend_upper = columns.Double()
    update_date = columns.DateTime()
    valuation_code = columns.Text()
    windowing_timestamp = columns.BigInt()
    work_center_id = columns.Integer()
    work_center_name = columns.Text()
    work_center_no = columns.Text()
    work_station_id = columns.Integer()
    work_station_name = columns.Text()
    work_station_no = columns.Text()
    workstation_state = columns.Text()
    year = columns.Text()
    sensor_values = columns.Map(columns.Text, columns.Map(columns.Text, columns.Text))

    @classmethod
    def saveData(cls, message, main_message, anomaly, key, upper, lower, sensor_dict, table_name):
        ph2_logger.debug(f"saveData - Processing data for table {table_name}")
        ph2_logger.debug(f"saveData - Input message: {json.dumps(message, default=str)}")
        ph2_logger.debug(f"saveData - Input main_message: {json.dumps(main_message.__dict__ if hasattr(main_message, '__dict__') else main_message, default=str)}")
        ph2_logger.debug(f"saveData - Input anomaly: {anomaly}, key: {key}, upper: {upper}, lower: {lower}, sensor_dict: {sensor_dict}")

        table_columns = {
            "tbl_p2_shiftly_anomaly": {
                "primary_keys": ["year_month", "partition_date", "measurement_date", "unique_code"],
                "required": ["shift_start_text", "shift_start_time", "shift_finish_text", "shift_finish_time"]
            },
            "tbl_p2_one_minute_anomaly": {
                "primary_keys": ["year_month", "partition_date", "measurement_date", "unique_code"],
                "required": []
            },
            "tbl_p2_ten_minute_anomaly": {
                "primary_keys": ["year_month", "partition_date", "measurement_date", "unique_code"],
                "required": []
            },
            "tbl_p2_one_day_anomaly": {
                "primary_keys": ["year_month", "partition_date", "measurement_date", "unique_code"],
                "required": []
            },
            "tbl_p2_one_hour_anomaly": {
                "primary_keys": ["year_month", "partition_date", "measurement_date", "unique_code"],
                "required": []
            },
            "tbl_p2_one_week_anomaly": {
                "primary_keys": ["year_month", "partition_date", "measurement_date", "unique_code"],
                "required": []
            }
        }

        if table_name not in table_columns:
            ph2_logger.error(f"saveData - Invalid table_name: {table_name}")
            raise ValueError(f"Invalid table_name: {table_name}")

        valid_columns = table_columns[table_name]
        ph2_logger.debug(f"saveData - Valid columns for {table_name}: {valid_columns}")

        def parse_meas_dt(meas_dt, default=None):
            if meas_dt is None or meas_dt == '':
                ph2_logger.warning(f"parse_meas_dt - Missing or empty measDt, using default: {default}")
                return default
            try:
                # Handle string, int, or float
                if isinstance(meas_dt, (int, float)):
                    timestamp = meas_dt
                else:
                    timestamp = float(meas_dt)
                return datetime.fromtimestamp(timestamp / 1000, tz=pytz.UTC)
            except (ValueError, TypeError) as e:
                ph2_logger.warning(f"parse_meas_dt - Invalid measDt: {meas_dt}, error: {e}, using default: {default}")
                return default

        meas_dt = message.get("measDt")
        measurement_date = parse_meas_dt(meas_dt, default=datetime.now(pytz.UTC))
        year_month = measurement_date.strftime("%Y-%m")
        partition_date = measurement_date.strftime("%Y-%m-%d")
        month_year = f"{measurement_date.month}-{measurement_date.year}"
        year = message.get("year", measurement_date.strftime("%Y"))
        unique_code = message.get("unqCd", str(uuid.uuid4()))

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

        def safe_float(value, default=None):
            if value is None or value == '':
                return default
            try:
                return float(value)
            except (ValueError, TypeError):
                ph2_logger.warning(f"safe_float - Invalid float value: {value}, returning default: {default}")
                return default

        def safe_int(value, default=None):
            if value is None or value == '':
                return default
            try:
                return int(value)
            except (ValueError, TypeError):
                ph2_logger.warning(f"safe_int - Invalid int value: {value}, returning default: {default}")
                return default

        def to_datetime(ms, default=None):
            if ms is None or ms == '':
                return default
            try:
                return datetime.fromtimestamp(safe_int(ms, 0) / 1000, tz=pytz.UTC)
            except (ValueError, TypeError):
                ph2_logger.warning(f"to_datetime - Invalid timestamp: {ms}, returning default: {default}")
                return default

        counter_reading = safe_float(message.get("cntRead"), None)
        anomaly_score = safe_float(anomaly, None)
        anomaly_importance = safe_float(anomaly, None)
        detected_anomaly = safe_float(anomaly, None)
        bias = safe_float(message.get("bias"), 0.0)
        coefficient = safe_float(message.get("coef"), 0.0)
        mean_square_error = safe_float(message.get("msErr"), 0.0)
        predicted_lower = safe_float(lower, None)
        predicted_upper = safe_float(upper, None)
        predicted_value = safe_float(message.get("predicted_value"), 0.0)
        regression_value = safe_float(message.get("regVal"), 0.0)
        trend = safe_float(message.get("trend"), 0.0)
        trend_lower = safe_float(lower, None)
        trend_upper = safe_float(upper, None)

        values = {
            "year_month": year_month,
            "partition_date": partition_date,
            "measurement_date": measurement_date,
            "unique_code": unique_code,
            "active": bool(message.get("act")) if message.get("act") is not None else None,
            "anomaly": safe_int(anomaly, None),
            "anomaly_detection_active": bool(message.get("anomDetAct")) if message.get("anomDetAct") is not None else None,
            "anomaly_importance": anomaly_importance,
            "anomaly_score": anomaly_score,
            "bias": bias,
            "coefficient": coefficient,
            "counter_reading": counter_reading,
            "create_auto_maintenance_order_notification": bool(message.get("crAutoMaintOrdNot")) if message.get("crAutoMaintOrdNot") is not None else None,
            "create_date": to_datetime(message.get("crDt"), measurement_date),
            "customer": str(message.get("cust", "turna")) if message.get("cust") is not None else "turna",
            "detected_anomaly": detected_anomaly,
            "employee_id": safe_int(get_main_attr("employeeId", 0), None),
            "equipment_code_group_header_id": safe_int(message.get("eqCodeGrpHdrId", 0), None),
            "equipment_id": safe_int(message.get("eqId", 0), None),
            "equipment_measuring_point_id": safe_int(message.get("eqMeasPtId", 0), None),
            "equipment_name": str(message.get("eqNm", "")) if message.get("eqNm") is not None else None,
            "equipment_no": str(message.get("eqNo", "")) if message.get("eqNo") is not None else None,
            "id": uuid.uuid4(),
            "job_order_operation_id": safe_int(get_main_attr("jobOrderOperationId", 0), None),
            "machine_state": str(get_main_attr("machineState", "")) if get_main_attr("machineState") is not None else None,
            "max_error_count_in_hour": safe_int(message.get("maxErrCntHr", 0), None),
            "max_error_count_in_min": safe_int(message.get("maxErrCntMin", 0), None),
            "max_error_count_in_shift": safe_int(message.get("maxErrCntShft", 0), None),
            "mean_square_error": mean_square_error,
            "measurement_document_id": safe_int(message.get("measDocId", 0), None),
            "message_key": str(key) if key is not None else None,
            "month_year": str(message.get("monYr", month_year)) if message.get("monYr") is not None else month_year,
            "organization_id": safe_int(get_main_attr("organizationId", 0), None),
            "organization_name": str(get_main_attr("organizationName", "")) if get_main_attr("organizationName") is not None else None,
            "organization_no": str(get_main_attr("organizationNo", "")) if get_main_attr("organizationNo") is not None else None,
            "parameter": str(message.get("param", "")) if message.get("param") is not None else None,
            "plant_id": safe_int(get_main_attr("plantId", 0), None),
            "plant_name": str(get_main_attr("plantName", "")) if get_main_attr("plantName") is not None else None,
            "plant_no": str(get_main_attr("plantNo", "")) if get_main_attr("plantNo") is not None else None,
            "predicted_lower": predicted_lower,
            "predicted_upper": predicted_upper,
            "predicted_value": predicted_value,
            "produced_stock_id": safe_int(get_main_attr("producedStockId", 0), None),
            "produced_stock_name": str(get_main_attr("producedStockName", "")) if get_main_attr("producedStockName") is not None else None,
            "produced_stock_no": str(get_main_attr("producedStockNo", "")) if get_main_attr("producedStockNo") is not None else None,
            "regression_value": regression_value,
            "sensor_values": sensor_dict if isinstance(sensor_dict, dict) else {},
            "shift_finish_text": None,
            "shift_finish_time": None,
            "shift_start_text": None,
            "shift_start_time": None,
            "stock_id": safe_int(message.get("stockId", 0), None),
            "trend": trend,
            "trend_calculation_active": bool(message.get("trendCalcAct", False)) if message.get("trendCalcAct") is not None else None,
            "trend_lower": trend_lower,
            "trend_upper": trend_upper,
            "update_date": to_datetime(message.get("updDt"), None),
            "valuation_code": str(message.get("valCd", "")) if message.get("valCd") is not None else None,
            "windowing_timestamp": safe_int(meas_dt, int(measurement_date.timestamp() * 1000)),
            "work_center_id": safe_int(get_main_attr("workCenterId", 0), None),
            "work_center_name": str(get_main_attr("workCenterName", "")) if get_main_attr("workCenterName") is not None else None,
            "work_center_no": str(get_main_attr("workCenterNo", "")) if get_main_attr("workCenterNo") is not None else None,
            "work_station_id": safe_int(get_main_attr("workStationId", 0), None),
            "work_station_name": str(get_main_attr("workStationName", "")) if get_main_attr("workStationName") is not None else None,
            "work_station_no": str(get_main_attr("workStationNo", "")) if get_main_attr("workStationNo") is not None else None,
            "workstation_state": str(get_main_attr("workstationState", "")) if get_main_attr("workstationState") is not None else None,
            "year": year
        }

        if table_name == "tbl_p2_shiftly_anomaly":
            shift_start_time = get_main_attr("shiftStartTime", measurement_date)
            shift_finish_time = get_main_attr("shiftFinishTime", measurement_date)
            values.update({
                "shift_start_text": str(int(shift_start_time.timestamp() * 1000)) if shift_start_time else str(int(measurement_date.timestamp() * 1000)),
                "shift_start_time": shift_start_time,
                "shift_finish_text": str(int(shift_finish_time.timestamp() * 1000)) if shift_finish_time else str(int(measurement_date.timestamp() * 1000)),
                "shift_finish_time": shift_finish_time,
            })

        clean_values = {k: v for k, v in values.items() if v is not None}
        for req_field in valid_columns.get("required", []):
            if req_field not in clean_values:
                ph2_logger.warning(f"saveData - Missing required field {req_field} for {table_name}, setting to default")
                if req_field in ["shift_start_text", "shift_finish_text"]:
                    clean_values[req_field] = str(int(measurement_date.timestamp() * 1000))
                elif req_field in ["shift_start_time", "shift_finish_time"]:
                    clean_values[req_field] = measurement_date

        ph2_logger.debug(f"saveData - Saving to {table_name} with values: {json.dumps({k: str(v) for k, v in clean_values.items()}, indent=2)}")

        try:
            columns = ', '.join(clean_values.keys())
            placeholders = ', '.join(['%s'] * len(clean_values))
            query = f"INSERT INTO {keyspace}.{table_name} ({columns}) VALUES ({placeholders})"
            ph2_logger.debug(f"saveData - Executing CQL query: {query} with values: {list(clean_values.values())}")

            statement = SimpleStatement(query)
            session = connection.get_session()
            result = session.execute(statement, list(clean_values.values()))

            ph2_logger.debug(f"saveData - Successfully saved to {table_name}")
            return result
        except Exception as e:
            ph2_logger.error(f"saveData - Failed to save to {table_name}: {e}")
            raise

def save_anomaly_data2(message, main_message, anomaly, key, upper, lower, sensor_dict, granularity="one_day"):
    ph2_logger.debug(f"save_anomaly_data - Input: message={message}, main_message={main_message}, anomaly={anomaly}, key={key}, upper={upper}, lower={lower}, sensor_dict={sensor_dict}, granularity={granularity}")
    table_map = {
        "one_day": "tbl_p2_one_day_anomaly",
        "one_hour": "tbl_p2_one_hour_anomaly",
        "one_minute": "tbl_p2_one_minute_anomaly",
        "one_week": "tbl_p2_one_week_anomaly",
        "shiftly": "tbl_p2_shiftly_anomaly",
        "ten_minute": "tbl_p2_ten_minute_anomaly"
    }

    if granularity not in table_map:
        ph2_logger.error(f"save_anomaly_data - Invalid granularity: {granularity}")
        raise ValueError(f"Invalid granularity: {granularity}. Must be one of {list(table_map.keys())}")

    table_name = table_map[granularity]
    ph2_logger.debug(f"save_anomaly_data - Mapping granularity {granularity} to table {table_name}")

    try:
        data = phaseTwoReport.saveData(
            message=message,
            main_message=main_message,
            anomaly=anomaly,
            key=key,
            upper=upper,
            lower=lower,
            sensor_dict=sensor_dict,
            table_name=table_name
        )
        ph2_logger.debug(f"save_anomaly_data phase2 - Successfully saved to {table_name}")
        return data
    except Exception as e:
        ph2_logger.error(f"save_anomaly_data - Failed to save to {table_name}: {e}")
        raise



import uuid
from cassandra.cqlengine import columns
from cassandra.cqlengine.models import Model
from cassandra.cqlengine.management import sync_table
from cassandra.cqlengine import connection
from cassandra.auth import PlainTextAuthProvider
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

class DwTblAnomalies2(Model):
    __keyspace__ = keyspace
    __table_name__ = "dw_tbl_multiple_anomalies"
    year_month = columns.Text(primary_key=True)
    partition_date = columns.Text(primary_key=True, clustering_order="DESC")
    measurement_date = columns.DateTime(primary_key=True, clustering_order="DESC")
    unique_code = columns.Text(primary_key=True, clustering_order="DESC")
    active = columns.Boolean()
    anomaly = columns.Integer()
    anomaly_detection_active = columns.Boolean()
    anomaly_importance = columns.Double()
    bias = columns.Double()
    coefficient = columns.Double()
    counter_reading = columns.Double()
    create_auto_maintenance_order_notification = columns.Boolean()
    create_date = columns.DateTime()
    customer = columns.Text()
    employee_id = columns.Integer()
    equipment_code_group_header_id = columns.Integer()
    equipment_id = columns.Integer()
    equipment_measuring_point_id = columns.Integer()
    equipment_name = columns.Text()
    equipment_no = columns.Text()
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

    @classmethod
    def saveData(cls, message, main_message, y, key):
        if y == 0:
            upper = 1.0
            lower = -1.0
        else:
            upper = y + y * 0.05
            lower = y - y * 0.05
        reading = float(message.get("cntRead", 0)) if message.get("cntRead") else 0
        anomaly = 1 if reading > upper or reading < lower else 0

        def to_datetime(ms):
            return datetime.fromtimestamp(ms / 1000) if ms is not None else None

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

        partition_date = message.get("partition_date", datetime.now(pytz.UTC).strftime("%Y-%m-%d"))
        year_month = None
        if partition_date:
            try:
                date_obj = datetime.strptime(partition_date, "%Y-%m-%d")
                year_month = date_obj.strftime("%Y-%m")
            except (ValueError, TypeError):
                year_month = datetime.now(pytz.UTC).strftime("%Y-%m")

        meas_dt = message.get("measDt")
        measurement_date = to_datetime(int(meas_dt)) if meas_dt and meas_dt.isdigit() else datetime.now(pytz.UTC)

        values = {
            "year_month": year_month,
            "partition_date": partition_date,
            "measurement_date": measurement_date,
            "unique_code": str(message.get("unqCd", str(uuid.uuid4()))),
            "active": bool(message.get("act", False)),
            "anomaly": int(anomaly),
            "anomaly_detection_active": bool(message.get("anomDetAct", False)),
            "anomaly_importance": float(anomaly),
            "bias": float(message.get("bias", 0.0)) if message.get("bias") else 0.0,
            "coefficient": float(message.get("coef", 0.0)) if message.get("coef") else 0.0,
            "counter_reading": reading,
            "create_auto_maintenance_order_notification": bool(message.get("crAutoMaintOrdNot", False)),
            "create_date": get_main_attr("create_date", datetime.now(pytz.UTC)),
            "customer": str(message.get("cust", "turna")),
            "employee_id": int(get_main_attr("employee_id", 0)) if get_main_attr("employee_id") is not None else 0,
            "equipment_code_group_header_id": int(message.get("eqCodeGrpHdrId", 0)) if message.get("eqCodeGrpHdrId") else 0,
            "equipment_id": int(message.get("eqId", 0)) if message.get("eqId") else 0,
            "equipment_measuring_point_id": int(message.get("eqMeasPtId", 0)) if message.get("eqMeasPtId") else 0,
            "equipment_name": str(message.get("eqNm", "")),
            "equipment_no": str(message.get("eqNo", "")),
            "job_order_operation_id": int(get_main_attr("job_order_operation_id", 0)) if get_main_attr("job_order_operation_id") is not None else 0,
            "machine_state": str(get_main_attr("machine_state", "")),
            "max_error_count_in_hour": int(message.get("maxErrCntHr", 0)) if message.get("maxErrCntHr") else 0,
            "max_error_count_in_min": int(message.get("maxErrCntMin", 0)) if message.get("maxErrCntMin") else 0,
            "max_error_count_in_shift": int(message.get("maxErrCntShft", 0)) if message.get("maxErrCntShft") else 0,
            "mean_square_error": float(message.get("msErr", 0.0)) if message.get("msErr") else 0.0,
            "measurement_document_id": int(message.get("measDocId", 0)) if message.get("measDocId") else 0,
            "message_key": str(key),
            "month_year": str(message.get("monYr", year_month)),
            "organization_id": int(get_main_attr("organization_id", 0)) if get_main_attr("organization_id") is not None else 0,
            "organization_name": str(get_main_attr("organization_name", "")),
            "organization_no": str(get_main_attr("organization_no", "")),
            "parameter": str(message.get("param", "")),
            "plant_id": int(get_main_attr("plant_id", 0)) if get_main_attr("plant_id") is not None else 0,
            "plant_name": str(get_main_attr("plant_name", "")),
            "predicted_lower": float(lower),
            "predicted_upper": float(upper),
            "predicted_value": float(y),
            "produced_stock_id": int(get_main_attr("produced_stock_id", 0)) if get_main_attr("produced_stock_id") is not None else 0,
            "produced_stock_name": str(get_main_attr("produced_stock_name", "")),
            "produced_stock_no": str(get_main_attr("produced_stock_no", "")),
            "regression_value": float(message.get("regVal", 0.0)) if message.get("regVal") else 0.0,
            "shift_finish_text": str(int(get_main_attr("shift_finish_time", datetime.now(pytz.UTC)).timestamp() * 1000)),
            "shift_finish_time": get_main_attr("shift_finish_time", datetime.now(pytz.UTC)),
            "shift_start_text": str(int(get_main_attr("shift_start_time", datetime.now(pytz.UTC)).timestamp() * 1000)),
            "shift_start_time": get_main_attr("shift_start_time", datetime.now(pytz.UTC)),
            "stock_id": int(message.get("stockId", 0)) if message.get("stockId") else 0,
            "trend": float(y),
            "trend_calculation_active": bool(message.get("trendCalcAct", False)),
            "trend_lower": float(lower),
            "trend_upper": float(upper),
            "update_date": datetime.now(pytz.UTC),
            "valuation_code": str(message.get("valCd", "")),
            "windowing_timestamp": int(message.get("winTs", 0)) if message.get("winTs") else 0,
            "work_center_id": int(get_main_attr("work_center_id", 0)) if get_main_attr("work_center_id") is not None else 0,
            "work_center_name": str(get_main_attr("work_center_name", "")),
            "work_center_no": str(get_main_attr("work_center_no", "")),
            "work_station_id": int(get_main_attr("work_station_id", 0)) if get_main_attr("work_station_id") is not None else 0,
            "work_station_name": str(get_main_attr("work_station_name", "")),
            "work_station_no": str(get_main_attr("work_station_no", "")),
            "workstation_state": str(get_main_attr("workstation_state", "")),
            "year": str(message.get("year", datetime.now(pytz.UTC).strftime("%Y")))
        }

        clean_values = {k: v for k, v in values.items() if v is not None}
        return cls.create(**clean_values)

if __name__ == "__main__":
    sync_table(DwTblAnomalies2)
    
    

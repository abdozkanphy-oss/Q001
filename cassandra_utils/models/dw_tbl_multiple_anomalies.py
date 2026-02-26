import uuid
from cassandra.cqlengine import columns
from cassandra.cqlengine.models import Model
from cassandra.cqlengine.management import sync_table
from cassandra.cqlengine import connection
from cassandra.auth import PlainTextAuthProvider
from datetime import datetime
from utils.logger import ph3_logger
import pytz

# Cassandra Connection Setup
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

class DwTblMultipleAnomalies(Model):
    __keyspace__ = keyspace = cass.get("keyspace")

    __table_name__ = "dw_tbl_multiple_anomalies"
    year_month = columns.Text(primary_key=True)
    partition_date = columns.Text(primary_key=True, clustering_order="DESC")
    measurement_date = columns.DateTime(
        primary_key=True, clustering_order="DESC")
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
    sensor_values = columns.Map(
        columns.Text, columns.Map(columns.Text, columns.Text))
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
    def saveData(cls, *, message, main_message, anomaly, key, upper, lower, sensor_dict):
        
        def to_datetime(ms):
            return datetime.fromtimestamp(ms / 1000, tz=pytz.UTC) if ms else None

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
            "year_month":year_month,
            "partition_date":  partition_date,
            "unique_code": str(uuid.uuid4()),
            "measurement_date": measurement_date,
            "active": bool(message.get("act")) if message.get('act') else None,
            "anomaly_detection_active": bool(message.get("anomDetAct")) if message.get('anomDetAct') else None,
            "create_auto_maintenance_order_notification": bool(message.get("crAutoMaintOrdNot")) if message.get("crAutoMaintOrdNot") is not None else None,
            "anomaly": anomaly,
            "anomaly_importance": anomaly,
            "anomaly_score": anomaly,
            # "bias": float(message.get("bias")) if message.get("bias") else None,
            # "coefficient": float(message.get("coef")) if message.get("coef") else None,
            # "counter_reading": float(message.get("cntRead")) if message.get("cntRead") else None,
            "create_date": get_main_attr("create_date", datetime.now(pytz.UTC)),
            "detected_anomaly": float(anomaly), 
            "employee_id": int(main_message["employeeId"]) if main_message["employeeId"] is not None else None,
            "equipment_code_group_header_id": int(message.get("eqCodeGrpHdrId")) if message.get("eqCodeGrpHdrId") else None,
            "equipment_id": int(message.get("eqId")) if message.get("eqId") else None,
            "equipment_measuring_point_id": int(message.get("eqMeasPtId")) if message.get("eqMeasPtId") else None,
            "equipment_name": str(message.get("eqNm")) if message.get("eqNm") else None,
            "equipment_no": str(message.get("eqNo")) if message.get("eqNo") else None,
            # "id": uuid.uuid4(),
            "job_order_operation_id": int(main_message["jobOrderOperationId"]) if main_message["jobOrderOperationId"] else None,
            "machine_state": str(main_message["machineState"]) if main_message["machineState"] else None,
            # "max_error_count_in_hour": int(message.get("max_error_count_in_hour")) if message.get("max_error_count_in_hour") else None,
            # "max_error_count_in_min": int(message.get("max_error_count_in_min")) if message.get("max_error_count_in_min") else None,
            # "max_error_count_in_shift": int(message.get("max_error_count_in_shift")) if message.get("max_error_count_in_shift") else None,
            # "mean_square_error": float(message.get("mean_square_error")) if message.get("mean_square_error") else None,
            # "measurement_document_id": int(message.get("measurement_document_id")) if message.get("measurement_document_id") else None,
            "message_key": str(key) if key else None,
            "month_year": str(message.get("monYr")) if message.get("monYr") else None,
            "organization_id": int(main_message["organizationId"]) if main_message["organizationId"] else None,
            "organization_name": str(main_message["organizationName"]) if main_message["organizationName"] else None,
            "organization_no": str(main_message["organizationNo"]) if main_message["organizationNo"] else None,
            # "parameter": str(message.get("parameter")) if message.get("parameter") else None,
            
            "plant_id": int(main_message["plantId"]) if main_message["plantId"] else None,
            "plant_name": str(main_message["plantName"]) if main_message["plantName"] else None,
            # "plant_no": str(main_message["plantNo"]) if main_message["plantNo"] else None,
            "predicted_lower": float(lower),
            "predicted_upper": float(upper),
            # "predicted_value": float(y),
            "produced_stock_id": int(message.get("produced_stock_id")) if message.get("produced_stock_id") else None,
            "produced_stock_name": str(message.get("produced_stock_name")) if message.get("produced_stock_name") else None,
            "produced_stock_no": str(message.get("produced_stock_no")) if message.get("produced_stock_no") else None,
            "regression_value": float(message.get("regression_value")) if message.get("regression_value") else None,
            "sensor_values": message.get("sensor_values"),
            "shift_finish_text": str(int(main_message["shiftFinishTime"].timestamp() * 1000)) if main_message["shiftFinishTime"] else None,
            "shift_finish_time": main_message["shiftFinishTime"],
            "shift_start_text": str(int(main_message["shiftStartTime"].timestamp() * 1000)) if main_message["shiftStartTime"] else None,
            "shift_start_time": main_message["shiftStartTime"],
            # "stock_id": int(message.get("stock_id")) if message.get("stock_id") else None,
            # "trend": float(y),
            "trend_calculation_active": bool(message.get("trend_calculation_active")) if message.get("trend_calculation_active") is not None else None,
            "trend_lower": float(lower),
            "trend_upper": float(upper),
            "customer": str(message.get("cust")) if message.get("cust") else "turna",
            # "update_date": datetime.now(),
            # "valuation_code": str(message.get("valuation_code")) if message.get("valuation_code") else None,
            # "windowing_timestamp": int(message.get("windowing_timestamp")) if message.get("windowing_timestamp") else None,
            "work_center_id": int(main_message["workCenterId"]) if main_message["workCenterId"] else None,
            "work_center_name": str(main_message["workCenterName"]) if main_message["workCenterName"] else None,
            "work_center_no": str(main_message["workCenterNo"]) if main_message["workCenterNo"] else None,
            "work_station_id": int(main_message["workStationId"]) if main_message["workStationId"] else None,
            "work_station_name": str(main_message["workStationName"]) if main_message["workStationName"] else None,
            "work_station_no": str(main_message["workStationNo"]) if main_message["workStationNo"] else None,
            "workstation_state": str(message.get("workstation_state")) if message.get("workstation_state") else None,
            "year": str(message.get("year")) if message.get("year") else None,
            "sensor_values": sensor_dict
        }
        clean_values = {k: v for k, v in values.items() if v is not None}
        return cls.create(**clean_values)


if __name__ == "__main__":
    sync_table(DwTblMultipleAnomalies)
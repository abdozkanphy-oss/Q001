import uuid
import hashlib
from cassandra.cqlengine import columns
from cassandra.cqlengine import connection
from datetime import datetime
from cassandra.cqlengine.management import sync_table, drop_table
from cassandra.cqlengine.models import Model
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster
from utils.config_reader import ConfigReader

import logging
logger = logging.getLogger(__name__)

"""
UPDATED: Now supports both inVars (inputs) and outVals (outputs)
- equipment_type: True = INPUT, False = OUTPUT
- gen_read_val: input values (for inVars)
- counter_reading: output values (for outVals)
"""

cfg = ConfigReader()
cass = cfg.get("cassandra_props") or cfg.get("cassandra") or {}

cassandra_host = cass.get("host")
username = cass.get("username")
password = cass.get("password")
keyspace = cass.get("keyspace")

if not cassandra_host or not keyspace:
    raise RuntimeError(
        "Cassandra host/keyspace missing. Set in utils/config.json or via "
        "MSF_CASSANDRA_HOST and MSF_CASSANDRA_KEYSPACE."
    )

auth_provider = PlainTextAuthProvider(username=username, password=password) if username and password else None

def _parse_hosts(h):
    if isinstance(h, (list, tuple)):
        return [str(x).strip() for x in h if str(x).strip()]
    if isinstance(h, str):
        return [x.strip() for x in h.split(",") if x.strip()]
    return []

connection.setup(
    hosts=_parse_hosts(cassandra_host),
    default_keyspace=keyspace,
    auth_provider=auth_provider,
    protocol_version=4,
    lazy_connect=True,
    retry_connect=True,
    connect_timeout=20,
)


# --- small helpers for type conversion ---
def to_dt(ms_or_dt):
    if ms_or_dt is None:
        return None
    if isinstance(ms_or_dt, datetime):
        return ms_or_dt
    try:
        v = int(ms_or_dt)
        return datetime.utcfromtimestamp(v / 1000.0) if v > 10**11 else datetime.utcfromtimestamp(v)
    except Exception:
        return None

def to_int(x):
    try:
        return int(x)
    except Exception:
        return None

def to_float(x):
    try:
        return float(x)
    except Exception:
        return None

def to_bool(x):
    if isinstance(x, bool):
        return x
    if x in (0, 1):
        return bool(x)
    if isinstance(x, str):
        return x.strip().lower() in ("1", "true", "t", "yes", "y")
    return None

# --- unique code helpers ---
def choose_legacy_unique_code(top: dict, ov: dict) -> str:
    """
    Chooses a unique code from either top or ov dictionaries.

    Parameters:
        top (dict): top-level dictionary
        ov (dict): optional values dictionary

    Returns:
        str: a unique code
    """

    unq = None
    if isinstance(ov, dict):
        unq = ov.get("unqCd")

    if unq not in (None, "", "None"):
        return str(unq)

    return str(uuid.uuid4())



class dw_tbl_raw_data(Model):
    __keyspace__   = keyspace
    __table_name__ = "dw_tbl_raw_data"

    # PRIMARY KEY
    unique_code = columns.Text(partition_key=True)

    # === EXISTING COLUMNS ===
    active                                  = columns.Boolean()
    anomaly_detection_active                = columns.Boolean(default=False)
    good                                    = columns.Boolean()
    #bias                                    = columns.Double()
    #coefficient                             = columns.Double()
    counter_reading                         = columns.Double()  # For OUTPUT values
    #create_auto_maintenance_order_notification = columns.Boolean()
    create_date                             = columns.DateTime()
    customer                                = columns.Text()
    #employee_id                             = columns.Integer()
    #equipment_code_group_header_id          = columns.Integer()
    equipment_id                            = columns.Integer()
    #equipment_measuring_point_id            = columns.Integer()
    equipment_name                          = columns.Text()
    equipment_no                            = columns.Text()
    job_order_operation_id                  = columns.Integer()
    machine_state                           = columns.Text()
    #max_error_count_in_hour                 = columns.Integer()
    #max_error_count_in_min                  = columns.Integer()
    #max_error_count_in_shift                = columns.Integer()
    #mean_square_error                       = columns.Double()
    measurement_date                        = columns.DateTime()
    #measurement_document_id                 = columns.Integer()
    #month_year                              = columns.Text()
    operationname                           = columns.Text()
    operationno                             = columns.Text()
    operationtaskcode                       = columns.Text()
    #organization_id                         = columns.Integer()
    #organization_name                       = columns.Text()
    #organization_no                         = columns.Text()
    #parameter                               = columns.Text()
    plant_id                                = columns.Integer()
    plant_name                              = columns.Text()
    produced_stock_id                       = columns.Integer()
    produced_stock_name                     = columns.Text()
    produced_stock_no                       = columns.Text()
    #regression_value                        = columns.Double()
    shift_finish_time                       = columns.DateTime()
    shift_start_time                        = columns.DateTime()
    stock_id                                = columns.Integer()
    trend_calculation_active                = columns.Boolean()
    update_date                             = columns.DateTime()
    valuation_code                          = columns.Text()
    work_center_id                          = columns.Integer()
    work_center_name                        = columns.Text()
    work_center_no                          = columns.Text()
    work_station_id                         = columns.Integer()
    work_station_name                       = columns.Text()
    work_station_no                         = columns.Text()
    workstation_state                       = columns.Text()
    year                                    = columns.Text()
    job_order_reference_no                  = columns.Text()
    prod_order_reference_no                 = columns.Text()
    
    # === NEW COLUMNS for INPUT support ===
    gen_read_val                            = columns.Double()    # For INPUT values
    equipment_type                          = columns.Boolean()   # True = INPUT, False = OUTPUT

    @classmethod
    def saveData(cls, message: dict):
        """
        UPDATED: Save BOTH inVars (inputs) and outVals (outputs) to dw_tbl_raw_data.
        
        - outVals → equipment_type=False, counter_reading=value
        - inVars  → equipment_type=True, gen_read_val=value
        """
        top = message or {}
        
        # === PART 1: Save OUTPUT variables (outVals) ===
        outvals = top.get("outVals") or []
        if not isinstance(outvals, list):
            outvals = []

        prod_raw = top.get("prodList") or []
        if isinstance(prod_raw, dict):
            prod = prod_raw
        elif isinstance(prod_raw, list) and prod_raw and isinstance(prod_raw[0], dict):
            prod = prod_raw[0]
        else:
            prod = {}

        for ov in outvals:
            uniq = choose_legacy_unique_code(top, ov)
            
            row = {
                "unique_code": uniq,
                
                # NEW: Mark as OUTPUT
                "equipment_type": False,  # False = OUTPUT
                
                # OUTPUT value goes to counter_reading
                "counter_reading": to_float(ov.get("cntRead")),
                
                # gen_read_val stays None for outputs (safety)
                # "gen_read_val": None,  # Implicit
                
                # All other fields same as before
                "active": True, #to_bool(ov.get("act")),
                "anomaly_detection_active": to_bool(ov.get("anomDetAct")),
                #"bias": to_float(ov.get("bias")),
                #"coefficient": to_float(ov.get("coef")),
                #"create_auto_maintenance_order_notification": to_bool(ov.get("crAutoMaintOrdNot")),
                "create_date": to_dt(top.get("crDt")),
                "customer": (top.get("customer") or top.get("cust") or ov.get("cust")),
                #"employee_id": to_int(top.get("empId")),
                #"equipment_code_group_header_id": to_int(ov.get("eqCodeGrpHdrId")),
                "equipment_id": to_int(ov.get("eqId")),
                #"equipment_measuring_point_id": to_int(ov.get("eqMeasPtId")),
                "equipment_name": ov.get("eqNm"),
                "equipment_no": ov.get("eqNo"),
                "job_order_operation_id": to_int(top.get("joOpId")),
                "job_order_reference_no": str(top.get("joRef", "")),
                "prod_order_reference_no": str(top.get("refNo", "")),
                "good": to_bool(top.get("goodCnt")),
                "machine_state": top.get("mcSt"),
                #"max_error_count_in_hour": to_int(ov.get("maxErrCntHr")),
                #"max_error_count_in_min": to_int(ov.get("maxErrCntMin")),
                #"max_error_count_in_shift": to_int(ov.get("maxErrCntShft")),
                #"mean_square_error": to_float(ov.get("msErr")),
                "measurement_date": to_dt(ov.get("measDt")),
                #"measurement_document_id": to_int(ov.get("measDocId")),
                #"month_year": ov.get("monYr"),
                "operationname": top.get("opNm"),
                "operationno": top.get("opNo"),
                "operationtaskcode": top.get("opTc"),
                #"organization_id": to_int(top.get("orgId")),
                #"organization_name": top.get("orgNm"),
                #"organization_no": top.get("orgNo"),
                #"parameter": ov.get("param"),
                "plant_id": to_int(top.get("plId")),
                "plant_name": top.get("plNm"),
                "produced_stock_id": to_int(prod.get("stId")),
                "produced_stock_name": prod.get("stNm"),
                "produced_stock_no": prod.get("stNo"),
                #"regression_value": to_float(ov.get("regVal")),
                "shift_finish_time": to_dt(top.get("shFt")),
                "shift_start_time": to_dt(top.get("shSt")),
                "stock_id": to_int(prod.get("joStId")),
                "trend_calculation_active": to_bool(ov.get("trendCalcAct")),
                "update_date": to_dt(ov.get("updDt")),
                "valuation_code": ov.get("valCd"),
                "work_center_id": to_int(top.get("wcId")),
                "work_center_name": top.get("wcNm"),
                "work_center_no": top.get("wcNo"),
                "work_station_id": to_int(top.get("wsId")),
                "work_station_name": top.get("wsNm"),
                "work_station_no": top.get("wsNo"),
                "workstation_state": top.get("prSt"),
                "year": ov.get("year") or top.get("year"),
            }
            clean = {k: v for k, v in row.items() if v is not None}
            
            # Skip rows without timestamp
            if not clean.get("measurement_date"):
                continue
            
            cls.create(**clean)

            # ---- dual-write projection for training-friendly reads ----
            try:
                from cassandra_utils.models.dw_raw_by_ws import dw_tbl_raw_data_by_ws
                proj = {
                    "plant_id": clean.get("plant_id"),
                    "work_center_id": clean.get("work_center_id"),
                    "work_station_id": clean.get("work_station_id"),
                    "measurement_date": clean.get("measurement_date"),
                    "unique_code": clean.get("unique_code"),

                    "produced_stock_no": clean.get("produced_stock_no"),
                    "operationtaskcode": clean.get("operationtaskcode"),

                    "equipment_name": clean.get("equipment_name"),
                    "equipment_no": clean.get("equipment_no"),
                    "equipment_id": clean.get("equipment_id"),

                    "equipment_type": False,
                    "counter_reading": clean.get("counter_reading"),
                    "gen_read_val": None,

                    "customer": clean.get("customer"),
                }

                # M2.2 candidate groupers for context policy (dual-write into projection)
                proj["prod_order_reference_no"] = str(clean.get("prod_order_reference_no") or top.get("refNo") or "0")
                proj["job_order_reference_no"] = str(clean.get("job_order_reference_no") or top.get("joRef") or "0")

                jo_op = clean.get("job_order_operation_id") or top.get("joOpId") or 0
                try:
                    proj["job_order_operation_id"] = int(jo_op) if jo_op is not None else 0
                except Exception:
                    proj["job_order_operation_id"] = 0

                # keep False booleans; only drop None
                proj_clean = {k: v for k, v in proj.items() if v is not None or k == "equipment_type"}
                if proj_clean.get("measurement_date") and proj_clean.get("unique_code"):
                    dw_tbl_raw_data_by_ws.create(**proj_clean)

            except Exception as e:
                try:
                    logger.warning(f"[dualwrite_dw_by_ws] skipped due to error: {e}")
                except Exception:
                    pass


        # === PART 2: Save INPUT variables (inVars) ===
        # Support both field names
        invars = top.get("inVars") or top.get("inputVariableList") or []
        if not isinstance(invars, list):
            invars = []

        # === Handle rare top-level single reading ===
        if not outvals and not invars and (top.get("measDt") or top.get("cntRead")):
            # Safe synthetic ov for choose_legacy_unique_code (ov is not defined in this branch)
            _ov = {
                "unqCd": top.get("unqCd"),
            }
            uniq = choose_legacy_unique_code(top, _ov)

            row = {
                "unique_code": uniq,
                "equipment_type": False,  # Assume OUTPUT for top-level
                "counter_reading": to_float(top.get("cntRead")),
                "active": True, #to_bool(top.get("act")),
                "anomaly_detection_active": to_bool(top.get("anomDetAct")),
                "good": to_bool(top.get("goodCnt")),
                #"bias": to_float(top.get("bias")),
                #"coefficient": to_float(top.get("coef")),
                #"create_auto_maintenance_order_notification": to_bool(top.get("crAutoMaintOrdNot")),
                "create_date": to_dt(top.get("crDt")),
                "customer": (top.get("customer") or top.get("cust")),
                #"employee_id": to_int(top.get("empId")),
                #"equipment_code_group_header_id": to_int(top.get("eqCodeGrpHdrId")),
                "equipment_id": to_int(top.get("eqId")),
                #"equipment_measuring_point_id": to_int(top.get("eqMeasPtId")),
                "equipment_name": top.get("eqNm"),
                "equipment_no": top.get("eqNo"),
                "job_order_operation_id": to_int(top.get("joOpId")),
                "job_order_reference_no": str(top.get("joRef", "")),
                "prod_order_reference_no": str(top.get("refNo", "")),
                "machine_state": top.get("mcSt"),
                #"mean_square_error": to_float(top.get("msErr")),
                "measurement_date": to_dt(top.get("measDt")),
                #"measurement_document_id": to_int(top.get("measDocId")),
                #"month_year": top.get("monYr"),
                "organization_id": to_int(top.get("orgId")),
                "organization_name": top.get("orgNm"),
                "organization_no": top.get("orgNo"),
                #"parameter": top.get("param"),
                "plant_id": to_int(top.get("plId")),
                "plant_name": top.get("plNm"),
                #"regression_value": to_float(top.get("regVal")),
                "shift_finish_time": to_dt(top.get("shFt")),
                "shift_start_time": to_dt(top.get("shSt")),
                "trend_calculation_active": to_bool(top.get("trendCalcAct")),
                "update_date": to_dt(top.get("updDt")),
                "valuation_code": top.get("valCd"),
                "work_center_id": to_int(top.get("wcId")),
                "work_center_name": top.get("wcNm"),
                "work_center_no": top.get("wcNo"),
                "work_station_id": to_int(top.get("wsId")),
                "work_station_name": top.get("wsNm"),
                "work_station_no": top.get("wsNo"),
                "workstation_state": top.get("workstation_state"),
                "year": top.get("year"),
            }
            clean = {k: v for k, v in row.items() if v is not None}
            if clean.get("measurement_date"):
                cls.create(**clean)

        return message

    @classmethod
    def fetchData(cls, limit=60):
        """
        UPDATED: Returns both inputs and outputs.
        
        Returns:
            (rows, inputList, outputList, batchList)
            - inputList: list of input variable dicts (from inVars)
            - outputList: list of single-item lists (from outVals)
        """
        
        rows = list(cls.objects.limit(limit))
        input_list, batch_list, output_list = [], [], []

        for r in rows:
            # Check equipment_type to determine if INPUT or OUTPUT
            is_input = r.equipment_type if r.equipment_type is not None else False
            
            if is_input:
                # INPUT variable
                iv = {
                    "varNm": r.equipment_name,
                    "varNo": r.equipment_no,
                    "genReadVal": r.gen_read_val,
                    "measDt": int(r.measurement_date.timestamp() * 1000) if r.measurement_date else None,
                    "varId": r.equipment_id,
                    "act": r.active,
                    "joOpId": r.job_order_operation_id,
                    "plId": r.plant_id,
                    "wsId": r.work_station_id,
                    "wcId": r.work_center_id,
                    "good": r.good,
                    "eqId": r.equipment_id,
                    "cust": r.customer,
                    "joRef": r.job_order_reference_no,
                    "refNo": r.prod_order_reference_no,
                    "anomDetAct": r.anomaly_detection_active,
                    "trendCalcAct": r.trend_calculation_active
                }
                input_list.append(iv)
            else:
                # OUTPUT variable
                ov = {
                    "cntRead": r.counter_reading,
                    "measDt": int(r.measurement_date.timestamp() * 1000) if r.measurement_date else None,
                    #"valCd": r.valuation_code,
                    #"param": r.parameter,
                    "eqId": r.equipment_id,
                    #"eqMeasPtId": r.equipment_measuring_point_id,
                    #"eqCodeGrpHdrId": r.equipment_code_group_header_id,
                    "eqNm": r.equipment_name,
                    "eqNo": r.equipment_no,
                    "cust": r.customer,
                    "joRef": r.job_order_reference_no,
                    "refNo": r.prod_order_reference_no,
                    #"monYr": r.month_year,
                    #"measDocId": r.measurement_document_id,
                    "anomDetAct": r.anomaly_detection_active,
                    "trendCalcAct": r.trend_calculation_active,
                    #"msErr": r.mean_square_error,
                    #"coef": r.coefficient,
                    #"regVal": r.regression_value,
                    "plId": r.plant_id,
                    "wsId": r.work_station_id,
                    "wcId": r.work_center_id,
                    "good": r.good
                }
                output_list.append([ov])  # single-item list for compatibility

        return rows, input_list, output_list, batch_list


if __name__ == "__main__":
    sync_table(dw_tbl_raw_data)
import uuid
from cassandra.cqlengine import columns
from cassandra.cqlengine.models import Model
from datetime import datetime
##
import uuid
from cassandra.cqlengine import columns
from cassandra.cqlengine import connection
from datetime import datetime
from cassandra.cqlengine.management import sync_table, drop_table
from cassandra.cqlengine.models import Model
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster

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


def map_to_text(obj):
    if obj is None or not isinstance(obj, dict):
        return {}
    return {k: str(v) if v is not None else '' for k, v in obj.items()}


class dw_raw_data(Model):
    __keyspace__   = keyspace
    __table_name__ = "dw_raw_data"
    partition_key = columns.Text(primary_key=True, default="latest")
    measurement_date = columns.DateTime(primary_key=True, clustering_order="DESC")

    anomalydetectionactive = columns.Boolean()
    currentquantity = columns.Integer()
    good = columns.Boolean()
    joborderoperationid = columns.Integer()
    joborderoperationrefid = columns.Text()
    machinestate = columns.Text()
    operationname = columns.Text()
    operationno = columns.Text()
    operationtaskcode = columns.Text()
    outputstockid = columns.Integer()
    plantid = columns.Integer()
    plantname = columns.Text()
    prodorderoperationrefid = columns.Text()
    prodorderrefid = columns.Text()
    productionstate = columns.Text()
    quantitychanged = columns.Integer()
    shiftfinishtime = columns.DateTime()
    shiftid = columns.Integer()
    shiftname = columns.Text()
    shiftstarttime = columns.DateTime()
    unique_code = columns.UUID(default=uuid.uuid4)
    workcenterid = columns.Integer()
    workcentername = columns.Text()
    workcenterno = columns.Text()
    workstationid = columns.Integer()
    workstationname = columns.Text()
    workstationno = columns.Text()
    workstationstate = columns.Text()

    componentbatchlist = columns.List(columns.Map(columns.Text, columns.Text))
    inputvariablelist = columns.List(columns.Map(columns.Text, columns.Text))
    outputvaluelist = columns.List(columns.Map(columns.Text, columns.Text))
    producelist = columns.List(columns.Map(columns.Text, columns.Text))
    qualitychecklist = columns.List(columns.Map(columns.Text, columns.Text))

    @classmethod
    def saveData(cls, message):
        def ts(ms): return datetime.fromtimestamp(ms / 1000) if ms else None
        def maplist(lst): return [ {str(k): str(v) if v is not None else '' for k, v in d.items()} for d in lst or [] ]

        in_vars = message.get('inVars', []) or []
        comp_bats = message.get('compBats', []) or []
        out_vals = message.get('outVals', []) or []
        prod_list = message.get('prodList', []) or []
        qc_list = message.get('qCList', []) or []


        anom_det_act= pr_stk_id =None
        if isinstance(out_vals, list):
            anom_det_act = next((str(item.get("anomDetAct")).lower() == "true" for item in out_vals if "anomDetAct" in item), False)

        if isinstance(prod_list, list):
            pr_stk_id = next((item.get("stId") for item in prod_list if item.get("stId")), None)

        fields = {
            "partition_key": "latest",
            "unique_code": uuid.uuid4(),
            "measurement_date": ts(message.get('crDt')),
            "anomalydetectionactive": anom_det_act,
            "currentquantity": message.get("currCycQty"),
            "good": message.get("goodCnt"),
            "joborderoperationid": message.get("joOpId"),
            "joborderoperationrefid": message.get("joRef"),
            "machinestate": message.get("mcSt"),
            "operationname": message.get("opNm"),
            "operationno": message.get("opNo"),
            "operationtaskcode": message.get("opTc"),
            "outputstockid": pr_stk_id, #
            "plantid": message.get("plId"),
            "plantname": message.get("plNm"),
            "prodorderoperationrefid": None, #
            "prodorderrefid": None, #
            "productionstate": message.get("prSt"),
            "quantitychanged": message.get("chngCycQty"),
            "shiftfinishtime": ts(message.get("shFt")),
            "shiftid": message.get("shId"),
            "shiftname": message.get("shNm"),
            "shiftstarttime": ts(message.get("shSt")),
            "workcenterid": message.get("wcId"),
            "workcentername": message.get("wcNm"),
            "workcenterno": message.get("wcNo"),
            "workstationid": message.get("wsId"),
            "workstationname": message.get("wsNm"),
            "workstationno": message.get("wsNo"),
            "workstationstate": message.get("prSt"),

            "componentbatchlist": [map_to_text(batch) for batch in comp_bats],
            "inputvariablelist": [map_to_text(var) for var in in_vars],
            "outputvaluelist": [map_to_text(out) for out in out_vals],
            "producelist": [map_to_text(prod) for prod in prod_list],
            "qualitychecklist": [map_to_text(qc) for qc in qc_list]
        }

        # Strip out None values
        clean_fields = {k: v for k, v in fields.items() if v is not None}
        return cls.create(**clean_fields)

    @classmethod
    def fetchData(cls, limit=60):
        rows = cls.objects(partition_key = "latest").allow_filtering().order_by('-measurement_date').limit(limit)
        returnList = []
        outputList = []
        inputList = []
        batchList = []
    #    qualityList = []
        for row in rows:
            returnList.append(row)
            outputList.append(row['outputvaluelist']) 
            inputList.append(row['inputvariablelist'])
            batchList.append(row['producelist'])
            #qualityList.append(row[''])
        return returnList, inputList, outputList, batchList


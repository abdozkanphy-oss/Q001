import uuid
from cassandra.cqlengine import columns
from cassandra.cqlengine import connection
from datetime import datetime
from cassandra.cqlengine.management import sync_table
from cassandra.cqlengine.models import Model
from cassandra.auth import PlainTextAuthProvider
from cassandra.cqlengine import columns, connection, models
from cassandra.cluster import Cluster
import pytz
from utils.logger import ph3_1_logger

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


class scada_correlation_matrix(Model):
    __keyspace__ = keyspace
    partition_key = columns.Text(primary_key=True, default="latest")
    partition_date = columns.DateTime(
        primary_key=True, clustering_order="DESC")
    process_id = columns.Text(required=False)
    start_date = columns.DateTime(required=False)
    end_date = columns.DateTime(required=False)

    workstation_no = columns.Text(required=False)
    workstation_name = columns.Text(required=False)
    workcenter_no = columns.Text(required=False)
    workcenter_name = columns.Text(required=False)

    input_parameter_no = columns.Text(required=False)
    input_parameter_name = columns.Text(required=False)
    operator_no = columns.Text(required=False)
    operator_name = columns.Text(required=False)
    output_stock_no = columns.Text(required=False)
    output_stock_name = columns.Text(required=False)
    output_parameter_no = columns.Map(
        columns.Text, columns.Double, required=False)
    output_correlation_values = columns.Map(
        columns.Text, columns.Map(columns.Text, columns.Double), required=False)
    input_parameter_no = columns.Map(
        columns.Text, columns.Double, required=False)
    input_correlation_values = columns.Map(columns.Text, columns.Map(
        columns.Text, columns.Double), required=False)

    plant_id = columns.Text(required=False)
    customer = columns.Text(required=False)

    @classmethod
    def saveData(cls, *, message, main_message, correlation, outputDict, inputDict):
        values = {
            'partition_key': "latest",
            'partition_date': datetime.now(pytz.UTC),
            # 'process_id': 
            'start_date': main_message["shiftStartTime"], 
            "end_date": main_message["shiftFinishTime"],
            
            "workstation_no": str(main_message["workStationNo"]) if main_message["workStationNo"] else None,
            "workstation_name": str(main_message["workStationName"]) if main_message["workStationName"] else None,
            "workcenter_no": str(main_message["workCenterNo"]) if main_message["workCenterNo"] else None,
            "workcenter_name": str(main_message["workCenterName"]) if main_message["workCenterName"] else None,

            
            "input_parameter_no": inputDict,
            "output_parameter_no": outputDict,

            "output_correlation_values": correlation,
            "input_correlation_values": correlation,

            "plant_id": str(main_message["plantId"]) if main_message["plantId"] else None,
            "customer": str(message.get("cust")) if message.get("cust") else "turna",


    # input_parameter_name = columns.Text(required=False)
    # operator_name = columns.Text(required=False)
    # output_stock_no = columns.Text(required=False)
    # output_stock_name = columns.Text(required=False)

    # operator_no = columns.Text(required=False)
    # output_stock_no = columns.Text(required=False)
    # output_stock_name = columns.Text(required=False)
        }


        clean_values = {k: v for k, v in values.items() if v is not None}

        return cls.create(**clean_values)


class scada_mean_values(Model):
    __keyspace__ = keyspace
    partition_key = columns.Text(primary_key=True, default="latest")
    partition_date = columns.DateTime(
        primary_key=True, clustering_order="DESC")
    process_id = columns.Text(required=False)
    start_date = columns.DateTime(required=False)
    end_date = columns.DateTime(required=False)

    workstation_no = columns.Text(required=False)
    workstation_name = columns.Text(required=False)
    workcenter_no = columns.Text(required=False)
    workcenter_name = columns.Text(required=False)

    input_parameter_no = columns.Text(required=False)
    input_parameter_name = columns.Text(required=False)

    operator_no = columns.Text(required=False)
    operator_name = columns.Text(required=False)

    output_stock_no = columns.Text(required=False)
    output_stock_name = columns.Text(required=False)
    output_name = columns.Text(required=False)
    mean_value = columns.Map(columns.Text, columns.Double, required=False)

    plant_id = columns.Text(required=False)
    customer = columns.Text(required=False)

    @classmethod
    def saveData(cls, *, message, main_message, mean, outputText, inputText):
        values = {
            'partition_key': "latest",
            'partition_date': datetime.now(pytz.UTC),
            # 'process_id': 
            'start_date': main_message["shiftStartTime"], 
            "end_date": main_message["shiftFinishTime"],
            
            "workstation_no": str(main_message["workStationNo"]) if main_message["workStationNo"] else None,
            "workstation_name": str(main_message["workStationName"]) if main_message["workStationName"] else None,
            "workcenter_no": str(main_message["workCenterNo"]) if main_message["workCenterNo"] else None,
            "workcenter_name": str(main_message["workCenterName"]) if main_message["workCenterName"] else None,

            
            # "input_parameter_no": inputDict, # to be implemented
            # input_parameter_name = columns.Text(required=False)

            # operator_no = columns.Text(required=False)
            # operator_name = columns.Text(required=False)


            # output_stock_no = columns.Text(required=False)
            # output_stock_name = columns.Text(required=False)
            # output_name = columns.Text(required=False) # to be implemented

            "mean_value": mean,

            "plant_id": str(main_message["plantId"]) if main_message["plantId"] else None,
            "customer": str(message.get("cust")) if message.get("cust") else "turna",



        }


        clean_values = {k: v for k, v in values.items() if v is not None}

        return cls.create(**clean_values)

class scada_feature_importance_values(Model):
    __keyspace__ = keyspace
    partition_key = columns.Text(primary_key=True, default="latest")
    partition_date = columns.DateTime(
        primary_key=True, clustering_order="DESC")
    process_id = columns.Text(required=False)
    start_date = columns.DateTime(required=False)
    end_date = columns.DateTime(required=False)

    workstation_no = columns.Text(required=False)
    workstation_name = columns.Text(required=False)
    workcenter_no = columns.Text(required=False)
    workcenter_name = columns.Text(required=False)

    input_parameter_no = columns.Text(required=False)
    input_parameter_name = columns.Text(required=False)
    operator_no = columns.Text(required=False)
    operator_name = columns.Text(required=False)

    output_stock_no = columns.Text(required=False)
    output_stock_name = columns.Text(required=False)
    output_name = columns.Text(required=False)
    feature_importance_value = columns.Map(
        columns.Text, columns.Double, required=False)

    plant_id = columns.Text(required=False)
    customer = columns.Text(required=False)


    @classmethod
    def saveData(cls, *, message, main_message, feature_importance, outputText, inputText):
        values = {
            'partition_key': "latest",
            'partition_date': datetime.now(pytz.UTC),
            # 'process_id': 
            'start_date': main_message["shiftStartTime"], 
            "end_date": main_message["shiftFinishTime"],
            
            "workstation_no": str(main_message["workStationNo"]) if main_message["workStationNo"] else None,
            "workstation_name": str(main_message["workStationName"]) if main_message["workStationName"] else None,
            "workcenter_no": str(main_message["workCenterNo"]) if main_message["workCenterNo"] else None,
            "workcenter_name": str(main_message["workCenterName"]) if main_message["workCenterName"] else None,

            
            # "input_parameter_no": inputDict, # to be implemented
            # input_parameter_name = columns.Text(required=False)

            # operator_no = columns.Text(required=False)
            # operator_name = columns.Text(required=False)


            # output_stock_no = columns.Text(required=False)
            # output_stock_name = columns.Text(required=False)
            # output_name = columns.Text(required=False) # to be implemented

            "feature_importance_value": feature_importance,

            "plant_id": str(main_message["plantId"]) if main_message["plantId"] else None,
            "customer": str(message.get("cust")) if message.get("cust") else "turna",



        }


        clean_values = {k: v for k, v in values.items() if v is not None}

        return cls.create(**clean_values)

class scada_real_time_predictions(Model):
    __keyspace__ = keyspace
    partition_key = columns.Text(primary_key=True, default="latest")
    partition_date = columns.DateTime(
        primary_key=True, clustering_order="DESC")
    process_id = columns.Text(required=False)
    start_date = columns.DateTime(required=False)
    end_date = columns.DateTime(required=False)
    current = columns.DateTime(required=False)

    workstation_no = columns.Text(required=False)
    workstation_name = columns.Text(required=False)
    workcenter_no = columns.Text(required=False)
    workcenter_name = columns.Text(required=False)
    input_parameter_no = columns.Text(required=False)
    input_parameter_name = columns.Text(required=False)

    operator_no = columns.Text(required=False)
    operator_name = columns.Text(required=False)
    output_stock_no = columns.Text(required=False)
    output_stock_name = columns.Text(required=False)

    output_parameter_no = columns.List(columns.Text, required=False)
    output_predicted_values = columns.Map(columns.Text, columns.Map(
        columns.Text, columns.Map(columns.Text, columns.Double)), required=False)

    input_parameter_no = columns.List(columns.Text, required=False)
    input_predicted_values = columns.Map(columns.Text, columns.Map(
        columns.Text, columns.Map(columns.Text, columns.Double)), required=False)

    plant_id = columns.Text(required=False)
    customer = columns.Text(required=False)


def delete_model():
    try:
        session.set_keyspace(keyspace)

        session.execute("DROP TABLE IF EXISTS scada_mean_values;")
        session.execute("DROP TABLE IF EXISTS scada_correlation_matrix;")
        session.execute(
            "DROP TABLE IF EXISTS scada_feature_importance_values;")
        session.execute("DROP TABLE IF EXISTS scada_real_time_predictions;")
        print("Tables deleted successfully.")

    except Exception as e:
        print(f"Error deleting tables: {e}")
    finally:
        cluster.shutdown()




def close_cassandra_connection():
    session.shutdown()
    cluster.shutdown()


# delete_model()
# fake_scada_mean_values_list = [
#     {
#         "partition_key": "latest",
#         "partition_date": datetime(2025, 4, 15, 10 + i, 0),
#         "process_id": f"P002{i}",
#         "start_date": datetime(2025, 4, 10 + i, 8, 0),
#         "end_date": datetime(2025, 4, 14 + i, 17, 0),
#         "workstation_no": f"WS002{i}",
#         "workstation_name": f"Station {chr(65 + i)}",
#         "workcenter_no": f"WC002{i}",
#         "workcenter_name": f"Center {chr(65 + i)}",
#         "input_parameter_no": f"sensor_{10 + i}",
#         "input_parameter_name": f"Temperature Sensor {i}",
#         "operator_no": f"OP002{i}",
#         "operator_name": f"Operator {chr(65 + i)}",
#         "output_stock_no": f"ST02{i}",
#         "output_stock_name": f"Stock {chr(65 + i)}",
#         "output_name": f"Output {chr(65 + i)}",
#         "mean_value": {f"sensor_{10 + i}": 25.5 + i, f"sensor_{11 + i}": 30.2 + i},
#         "plant_id": f"PL02{i}",
#         "customer": f"Customer {chr(65 + i)}"
#     } for i in range(10)
# ]
# for fake_data in fake_scada_mean_values_list:
#     scada_mean_values.create(**fake_data)

# fake_scada_feature_importance_values_list = [
#     {
#         "partition_key": "latest",
#         "partition_date": datetime(2025, 4, 15, 10 + i, 0),
#         "process_id": f"P003{i}",
#         "start_date": datetime(2025, 4, 10 + i, 8, 0),
#         "end_date": datetime(2025, 4, 14 + i, 17, 0),
#         "workstation_no": f"WS003{i}",
#         "workstation_name": f"Station {chr(75 + i)}",
#         "workcenter_no": f"WC003{i}",
#         "workcenter_name": f"Center {chr(75 + i)}",
#         "input_parameter_no": f"sensor_{20 + i}",
#         "input_parameter_name": f"Pressure Sensor {i}",
#         "operator_no": f"OP003{i}",
#         "operator_name": f"Operator {chr(75 + i)}",
#         "output_stock_no": f"ST03{i}",
#         "output_stock_name": f"Stock {chr(75 + i)}",
#         "output_name": f"Output {chr(75 + i)}",
#         "feature_importance_value": {f"sensor_{20 + i}": 0.75 + i * 0.01, f"sensor_{21 + i}": 0.85 + i * 0.01},
#         "plant_id": f"PL03{i}",
#         "customer": f"Customer {chr(75 + i)}"
#     } for i in range(10)
# ]

# for fake_data in fake_scada_feature_importance_values_list:
#     scada_feature_importance_values.create(**fake_data)

# fake_scada_real_time_predictions_list = [
#     {
#         "partition_key": "latest",
#         "partition_date": datetime(2025, 3, 15+i, 8 + i, 0),  # different date pattern
#         "process_id": f"P008{i}",  # changed process ID pattern
#         "start_date": datetime(2025, 3, 1 + i, 9, 0),
#         "end_date": datetime(2025, 3, 4 + i, 16, 0),
#         "current": datetime(2025, 3, 25, 14 + i, 0),
#         "workstation_no": f"WS008{i}",
#         "workstation_name": f"Station {chr(65 + i)}",  # using different ASCII range
#         "workcenter_no": f"WC008{i}",
#         "workcenter_name": f"Center {chr(65 + i)}",
#         "input_parameter_no": f"sensor_{50 + i}",  # different sensor range
#         "input_parameter_name": f"Pressure Sensor {i}",  # changed sensor type
#         "operator_no": f"OP008{i}",
#         "operator_name": f"Operator {chr(65 + i)}",
#         "output_stock_no": f"ST08{i}",
#         "output_stock_name": f"Product {chr(65 + i)}",  # changed to Product instead of Stock
#         "output_parameter_no": [f"param_{60 + i}", f"{i+5}", f"{i+ 15}"],  # different parameter ranges
#         "output_predicted_values":{
#             f'sensor_{15+i}': {  # different sensor range
#                 'Temperature': { 
#                     'predicted_value': f"{120.25 + i}"  # different base value
#                 },
#                 'Pressure': {
#                     'predicted_value': f"{1.75 + i}"
#                 }
#             },
#             f'sensor_{15+i}': {
#                 'FlowRate': {
#                     'predicted_value': f"{45.50 + i}"
#                 }
#             }
#         },
#         "input_parameter_no": [f"param_{30 + i}", f"{i+2}", f"{i+ 8}"],  # different parameter ranges
#         "input_predicted_values": {
#             f'sensor_{25+i}': {  # different sensor range
#                 'Temperature': { 
#                     'predicted_value': f"{55.25 + i}"  # different base values
#                 },
#                 'Pressure': {
#                     'predicted_value': f"{3.45 + i}"
#                 }
#             },
#             f'sensor_{25+i}': {
#                 'FlowRate': {
#                     'predicted_value': f"{42.25 + i}"
#                 },
#                 'Pressure': {
#                     'predicted_value': f"{3.45 + i}"
#                 }
#             }
#         },
#         "plant_id": f"PL08{i}",  # changed plant ID pattern
#         "customer": f"Industry {chr(65 + i)}"  # changed to Industry instead of Customer
#     } for i in range(10)
# ]

# for fake_data in fake_scada_real_time_predictions_list:
#     scada_real_time_predictions.create(**fake_data)

# sync_table(scada_mean_values)
# sync_table(scada_correlation_matrix)
# sync_table(scada_feature_importance_values)
# sync_table(scada_real_time_predictions)



from modules.kafka_modules import kafka_consumer3
import json
from utils.logger import ph3_logger
from cassandra_utils.models.dw_single_data import dw_single_data_data
from thread.phase3 import Phase3
from thread.phase1 import Phase1
from datetime import datetime
import pytz

# Configuration flag to control whether to process only current-date data
# Set to False to process old data (e.g., May 20, 21, 22, etc.)
PROCESS_CURRENT_DATE_ONLY = True

consumer3 = kafka_consumer3()

def consumer_preprocess_3():
    phase1_class = Phase1()
    phase3_class = Phase3()
    try:
        returnList, inputList, outputList, batchList = dw_single_data_data.fetchData(20_000)
        if not returnList:
            ph3_logger.warning("message_processor_3 - No data found in fetch call")
            ph3_logger.error(f"No data received for training")
            raise
        else:
            ph3_logger.debug("message_processor_3 - Data to be added to the output_training_dict")
            for i in range(len(outputList)):
                try:
                    phase1_class.add_output_training(outputList[i])
                except:
                    ph3_logger.error(f"message_processor_3 - training data addition error at index {i} totalLen {len(outputList)} data{outputList[i]}")
            ph3_logger.debug("message_processor_3 - Data added")
    except Exception as e:
        ph3_logger.error(f"message_preprocessor_3 - Error {e}")

    try:
        while True:
            msg = consumer3.poll(1.0)
            if msg is None:
                # ph3_logger.debug("consumer_preprocess_3 - No message received")
                continue  
            if msg.error():
                ph3_logger.error(f"consumer_preprocess_3 - Consumer error: {msg.error()}")
                continue
            try:
                raw_value = msg.value().decode('utf-8')
                message = json.loads(raw_value)
                # ph3_logger.debug(f"consumer_preprocess_3 - Message received: {message}")
                
                # Ensure inputVariableList is a list to avoid errors in Phase3
                if 'inputVariableList' not in message or message['inputVariableList'] is None:
                    message['inputVariableList'] = []
                    # ph3_logger.debug("consumer_preprocess_3 - Set inputVariableList to empty list")

                if isinstance(message, list):
                    # ph3_logger.error(f"consumer_preprocess_3 - The input data is a list")
                    continue

                # Proceed with processing the message
                data = dw_single_data_data.saveData(message)
                ph3_logger.debug(f"consumer_preprocess_3 - The data is saved in cassandra {data}")
                phase1_class.add_allData_predict(data)
                phase1_class.add_output_predict(data, data['outputValueList'])
                phase3_class.add_data(message)

                # Commit the offset to mark this message as processed
                consumer3.commit()
                ph3_logger.debug(f"consumer_preprocess_3 - Offset committed for message: {msg.offset()}")

            except Exception as e:
                ph3_logger.error(f"consumer_preprocess_3 - Error in Consumer data decode: {str(e)}")
                continue
    except Exception as e:
        ph3_logger.error(f"consumer_preprocess_3 - Error in consumer main function: {str(e)}")
    finally:
        # Clean up resources
        phase3_class.cleanup()
        consumer3.close()
        ph3_logger.info("consumer_preprocess_3 - Consumer closed and resources cleaned up")
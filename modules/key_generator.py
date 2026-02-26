
def main_key_generation(message):
    key = str(message['organizationName']) + "_" + str(message['plantId']) + "_" +  str(message['workCenterId']) + "_" + str(message['workStationId']) 
    return key

def output_key_generation(output, main = None, prefix = False):
    if main is None:
        # key = str(output['eqId']) + "_"  + str(output['eqNm']).replace(" ", "_") + "_" + str(output['eqNo'])
        key = str(output['eqId']) + "_"  + str(output['eqNm']).replace(" ", "_") + "_" + str(output['eqNo'])
    else:
        key = main_key_generation(main) + "_" + str(output['eqId']) + "_"  + str(output['eqNm']).replace(" ", "_")
    
    if prefix:
        key =  "output_" + key
    return key

def input_key_generation(input, main = None, prefix = False):
    if main is None:
        key = str(input['varId']) + "_" + str(input['varNm']).replace(" ", "_")
    else:
        key = main_key_generation(main) + "_" + str(input['varId']) + "_" + str(input['varNm']).replace(" ", "_")
    
    if prefix:
        key = "input_" + key
    return key

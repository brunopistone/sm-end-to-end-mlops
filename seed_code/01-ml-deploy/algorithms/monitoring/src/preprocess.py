import ast
import json

LABEL = 'Sentiment'

def write_to_file(log, filename):
    with open(f"/opt/ml/processing/output/{filename}.log", "a") as f:
        f.write(log + '\n')

def get_class_val(probability):
    v = ast.literal_eval(probability)
    return v["prediction"]

def preprocess_handler(inference_record):
    input_enc_type = inference_record.endpoint_input.encoding
    input_data = json.loads(inference_record.endpoint_input.data.rstrip("\n"))

    input_data["text"] = input_data["features"][0]

    del input_data["features"]

    input_data = json.dumps(input_data)
    output_data = get_class_val(inference_record.endpoint_output.data.rstrip("\n").rstrip("\n"))

    if input_enc_type == "JSON":
        outputs = {**{LABEL: output_data}, **json.loads(input_data)}
        write_to_file(str(outputs), "log")
        return {str(i).zfill(20): outputs[d] for i, d in enumerate(outputs)}
    else:
        raise ValueError(f"encoding type {input_enc_type} is not supported")

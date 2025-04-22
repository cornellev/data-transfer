import io
import avro.schema
import avro.io

# Loads and parses an Avro schema from a .avsc (json) file
def load_avro_schema(path):
    with open(path, "r") as f: 
        return avro.schema.parse(f.read())
    
# Encodes data using an Avro schema into bytes 
def encode_data (data, schema_path):
    schema = load_avro_schema(schema_path)
    writer = avro.io.DatumWriter(schema)

    bytes_writer = io.BytesIO()
    encoder = avro.io.BinaryEncoder(bytes_writer)

    writer.write(data, encoder)
    return bytes_writer.getvalue()


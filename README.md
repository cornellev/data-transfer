# Data Transfer Protocols
A repository for managing effective, efficient transmission of data from the vehicle to the control operations team for competition environments. We are planning to use the same protocol of voice calls, SMS, and StarLink.

Raspi Sender: When data is transferred in (via a POST to our Flask REST API), we encode it using Apache Avro (chose this due to its dynamic nature), transform that into audio wav's, send through whatever protocol we designate. Each requires a different feature and thus file. Inside the sender folder

- `restapi.py` - the flask api for receiving our data
- `data-encoder.py` - dealing with avro encodings for various data types from raspi
- `telephone/audio.py` - doing the telephone voice call stuff (just the different schemes of encodings)
- `telephone/sender.py` - pushing the voice call data


Laptop Receiver: When we receive data, we need to decode it and make a websocket endpoint for the frontend to subscribe to and display received + decoded data. Same features but it is a websocket instead of POST.

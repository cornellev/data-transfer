# Data Transfer Protocols
A repository for managing effective, efficient transmission of data from the vehicle to the control operations team for competition environments. We are planning to use the same protocol of voice calls, SMS, and StarLink.

Raspi Sender: When data is transferred in (via a POST to our Flask REST API), we encode it using Apache Avro (chose this due to its dynamic nature), transform that into audio wav's, send through whatever protocol we designate. Each requires a different feature and thus file.

Laptop Receiver: When we receive data, we need to decode it.
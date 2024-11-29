import zlib
import json
import logging
import base64

class DataCompressor:
    """
    A class for performing lossless compression and decompression of data.
    """

    @staticmethod
    def compress(data):
        """
        Compresses the given data using zlib and encodes it for JSON serialization.
        
        :param data: Data to be compressed (list or JSON-serializable object).
        :return: Base64-encoded compressed string or None if an error occurs.
        """
        try:
            # Convert data to a JSON string
            json_data = json.dumps(data)
            # Compress the JSON string
            compressed_data = zlib.compress(json_data.encode('utf-8'))
            # Encode compressed data to Base64 for JSON serialization
            base64_encoded_data = base64.b64encode(compressed_data).decode('utf-8')
            return base64_encoded_data
        except Exception as e:
            logging.error(f"Error compressing data: {e}")
            return None

    @staticmethod
    def decompress(encoded_data):
        """
        Decodes and decompresses Base64-encoded zlib-compressed data.
        
        :param encoded_data: Base64-encoded compressed string.
        :return: Original uncompressed data or None if an error occurs.
        """
        try:
            # Decode Base64 string to bytes
            compressed_data = base64.b64decode(encoded_data)
            # Decompress data
            decompressed_data = zlib.decompress(compressed_data).decode('utf-8')
            # Convert JSON string back to Python object
            return json.loads(decompressed_data)
        except Exception as e:
            logging.error(f"Error decompressing data: {e}")
            return None

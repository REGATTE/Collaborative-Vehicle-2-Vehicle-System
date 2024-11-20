import socket
import logging
import json

def start_ego_listener(host='127.0.0.1', port=65432):
    """
    Starts a listener on the ego vehicle to receive data from smart vehicles.
    :param host: Host address to listen on.
    :param port: Port to listen on.
    """
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen()
    logging.info(f"Ego Vehicle listening on {host}:{port}")

    while True:
        conn, addr = server_socket.accept()
        try:
            data = conn.recv(1024)
            if data:
                smart_data = json.loads(data.decode())
                logging.info(f"Received data from Smart Vehicle {smart_data['id']}: {smart_data}")
        except Exception as e:
            logging.error(f"Error receiving data: {e}")
        finally:
            conn.close()

import json
import socket
import threading
from utils.logger import log

class Server:
    def __init__(self, host='localhost', port=5001):
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(128)
        self.clients = []
        log(f"Server Started on {host}:{port}")

    def accept_clients(self):
        while True:
            try:
                client_socket, address = self.server_socket.accept()
                self.clients.append(client_socket)
                log(f"New Client Connected from {address}, {len(self.clients)} Clients Online")
            except Exception as e:
                log(f"Error Accepting Client: {e}")
                break

    def send_to_client(self, client_socket, data):
        try:
            message = json.dumps(data).encode('utf-8')
            message_length = len(message)
            client_socket.send(message_length.to_bytes(4, 'big'))
            client_socket.send(message)
        except Exception as e:
            log(f"Error Sending to Client: {e} data: {data}")

    def broadcast(self, string_array):
        data = {
            "type": "broadcast",
            "data": string_array
        }
        clients_copy = self.clients.copy()
        for client in clients_copy:
            self.send_to_client(client, data)
        log(f"Broadcasted {len(string_array)} prompts to {len(clients_copy)} Clients")

    def start(self):
        accept_thread = threading.Thread(target=self.accept_clients)
        accept_thread.daemon = True
        accept_thread.start()

    def close(self):
        for client in self.clients:
            try:
                client.close()
            except:
                pass
        self.server_socket.close()

class Client:
    def __init__(self, host='localhost', port=5001):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connected = False
        self.receive_thread = None

    def connect(self):
        try:
            self.socket.connect((self.host, self.port))
            self.connected = True
            return True
        except Exception as e:
            log(f"Connection Failed: {e}")
            return False

    def receive_messages(self, callback_func):
        while self.connected:
            try:
                length_bytes = self.socket.recv(4)
                if not length_bytes:
                    break
                message_length = int.from_bytes(length_bytes, 'big')
                message_bytes = self.socket.recv(message_length)
                if not message_bytes:
                    break
                message = json.loads(message_bytes.decode('utf-8'))
                if message["type"] == "connection_ack":
                    log(f"Server Message: {message['message']}")
                elif message["type"] == "broadcast":
                    callback_func(message['data'])
            except Exception as e:
                log(f"Error Receiving Message: {e}")
        self.connected = False
        log("Disconnected from Server")

    def close(self):
        self.connected = False
        try:
            self.socket.close()
        except:
            pass

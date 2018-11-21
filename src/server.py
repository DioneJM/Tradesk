import socket
import json

class Server:
    """ Interface for Server to communicate Java App """

    def __init__(self, hostname, port):

        print("Initialising Server\n")
        self.serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.serverSocket.bind((hostname, port))

        print("Listening on port {}".format(port))
        self.serverSocket.listen(10)
        self.connection, _ = self.serverSocket.accept()

    def receive_data(self):
        """ Receives json string from socket """
        j = ""
        if self.serverSocket:
            try:
                # while 1:
                message = self.connection.recv(1024)
                if message:
                    print("Received data from client")
                    j = json.loads(message)
                    print ("Got: {0}".format(j["name"]))
            except:
                raise
        return j

    def send_data(self, data):
        """ Send json string to socket """
        if self.connection and data:
            try:
                print("Sending data")
                self.connection.sendall(data)
            except:
                raise

    def shutdownServer(self):
        self.connection.close()

# Sample Code for setting up server
# s = Server("localhost", 2000)
#
# while True:
#     live_stream_data = s.receive_data()
#     if live_stream_data:
#         if "SIGEND PROGRAM" in live_stream_data:
#             s.shutdownServer()
#         s.send_data(get_predictions(testing[:sequence_size]), visualize=False)
#     else:
#         raise ValueError("No data received from stream\n")



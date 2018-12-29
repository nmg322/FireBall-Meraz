import time
import threading
import socket

import numpy as np
import cv2


class Stopper:
    def __init__(self):
        self.status = False

    def set(self):
        self.status = True
    
    def clear(self):
        self.status = False

    def is_set(self):
        return self.status


class WebcamVideoStream(threading.Thread):
    """
    Class to receive images from server.
    """

    def __init__(self, server_address):
        threading.Thread.__init__(self)

        self.stopper = Stopper()
        self.lock = threading.Lock()
        self.frame = None
        self.server_address = server_address
        # Create a UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(0.5)

        sent = self.sock.sendto(b"START", self.server_address)
        time.sleep(2)
        try:
            data, server = self.sock.recvfrom(30)
            if len(data)<20:
                self.cam_size = tuple(map(int, str(data, "utf-8").split()))
            else:
                print("Cannot decode camera resolution")
                self.cam_size = (240, 640)
        except socket.timeout:
            print("Cannot get camera resolution from server: Time Out")
            self.cam_size = (240, 640)

    def run(self):
        # Keep looping infinitely until the thread is stopped
        while not self.stopper.is_set():
            sent = self.sock.sendto(b"GET", self.server_address)
            try:
                data, server = self.sock.recvfrom(65507)
                if data == b"FAIL":
                    print("FAIL")
                elif len(data)>100:
                    array = np.frombuffer(data, dtype=np.dtype('uint8'))
                    self.lock.acquire()
                    self.frame = cv2.imdecode(array, 1)
                    self.lock.release()
                else:
                    print("Cannot decode image")
            except socket.timeout:
                print("Receive Timeout")

    def read(self):
        # Method to access the decoded frame.
        if self.frame is not None:
            self.lock.acquire()
            cpy = self.frame.copy()
            self.lock.release()
            return cpy

    def stop(self):
        # Indicate that the thread should be stopped
        self.stopper.set()
        sent = self.sock.sendto(b"STOP", self.server_address)

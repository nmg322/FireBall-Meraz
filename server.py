# For debugging :
# - run the server and remember the IP of the server
# And interact with it through the command line:
# echo -n "GET" > /dev/udp/192.168.0.39/1080
# echo -n "QUIT" > /dev/udp/192.168.0.39/1080

import socket
import cv2
import threading
import sys
import signal

if(len(sys.argv) != 2):
    print("Usage : {} host".format(sys.argv[0]))
    print("e.g. {} 127.0.0.1".format(sys.argv[0]))
    sys.exit(-1)

DEVICE_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
JPEG_QUALITY = 50
HOST = sys.argv[1]
PORT = 1080


class Stopper:
    def __init__(self):
        self.status = False

    def set(self):
        self.status = True
    
    def clear(self):
        self.status = False

    def is_set(self):
        return self.status


class SignalHandler:
    """
    The object that will handle signals and stop the worker threads.
    """

    def __init__(self, stopper, grabber, sock):
        self.stopper = stopper
        self.grabber = grabber
        self.sock = sock

    def __call__(self, signum, frame):
        """
        This will be called by the python signal module
        https://docs.python.org/3/library/signal.html#signal.signal
        """

        self.stopper.set()
        self.grabber.join()
        self.sock.close()
        cv2.destroyAllWindows()
        exit(0)


class VideoGrabber(threading.Thread):
    """A threaded video grabber.
    Attributes:
    encode_params (): 
    cap (str): 
    attr2 (:obj:'int', optional): Description of 'attr2'.
    """

    def __init__(self, jpeg_quality, device_index, camera_width, camera_height, stopper):
        """Constructor.
        Args:
        jpeg_quality (:obj:'int'): Quality of JPEG encoding, in 0, 100.
        """

        threading.Thread.__init__(self)
        self.encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        self.cap = cv2.VideoCapture(device_index)

        # Increase the resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
        print("Camera Resolution:", self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT), self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        self.stopper = stopper
        self.buffer = None
        self.lock = threading.Lock()

    def get_buffer(self):
        """Method to access the encoded buffer.
        Returns:
        np.ndarray: the compressed image if one has been acquired. None otherwise.
        """

        if self.buffer is not None:
            self.lock.acquire()
            cpy = self.buffer.copy()
            self.lock.release()
            return cpy

    def run(self):
        # Keep looping infinitely until the thread is stopped
        while not self.stopper.is_set():
            success, img = self.cap.read()
            if not success:
                continue
            # JPEG compression
            # Protected by a lock as the main thread may asks to access the buffer
            self.lock.acquire()
            result, self.buffer = cv2.imencode('.jpg', img, self.encode_param)
            self.lock.release()
        self.cap.release()


sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Bind the socket to the port
server_address = (HOST, PORT)

print("Server is running at %s:%s\n" % server_address)

sock.bind(server_address)

while True:
    data, address = sock.recvfrom(10)

    if data == b"START":
        stopper = Stopper()
        grabber = VideoGrabber(JPEG_QUALITY, DEVICE_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, stopper)
        grabber.start()
        handler = SignalHandler(stopper, grabber, sock)
        signal.signal(signal.SIGINT, handler)

        sock.sendto((str(int(grabber.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))+' '+\
                str(int(grabber.cap.get(cv2.CAP_PROP_FRAME_WIDTH)))).encode("utf-8"), address)

        while not stopper.is_set():
            data, address = sock.recvfrom(10)

            if data == b"GET":
                buffer = grabber.get_buffer()
                if buffer is None:
                    continue
                if len(buffer) > 65507:
                    print("The message is too large(", len(buffer), ")to be sent within a single UDP datagram.",
                    "We do not handle splitting the message in multiple datagrams")
                    sock.sendto(b"FAIL", address)
                    continue
                # We sent back the buffer to the client
                sock.sendto(buffer.tobytes(), address)
            elif data == b"STOP":
                stopper.set()
                grabber.join()
            # elif data == b"CAM_SIZE":
            #     sock.sendto((str(int(grabber.cap.get(cv2.CAP_PROP_FRAME_WIDTH)))+' '+\
            #     str(int(grabber.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))).encode("utf-8"), address)
    elif data == b"QUIT":
        break

print("Quitting...")
grabber.join()
sock.close()

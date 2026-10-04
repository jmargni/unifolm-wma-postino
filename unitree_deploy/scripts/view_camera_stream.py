import zmq, cv2, numpy as np                                                                                        
s = zmq.Context().socket(zmq.SUB); s.connect('tcp://127.0.0.1:5555'); s.setsockopt_string(zmq.SUBSCRIBE, '')
while cv2.waitKey(1) != 27:
    cv2.imshow('G1 cameras', cv2.imdecode(np.frombuffer(s.recv(), np.uint8), cv2.IMREAD_COLOR))


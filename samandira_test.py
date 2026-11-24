from sensors.camera_node import CameraNode
import time

cam = CameraNode(enable_preview=True)

for i in range(2000):
    d = cam.process_frame()
    if i % 10 == 0:
        print(d)
    time.sleep(0.03)

cam.shutdown()

import time
from msl.loadlib import Client64
class ATFClient(Client64):
    def __init__(self):
        super().__init__(module32='atf_server')
    def connect(self,com_port):
        return self.request32('connect',com_port)
    def disconnect(self):
        return self.request32('disconnect')
    
atf = ATFClient()
atf.connect('COM9')
time.sleep(5)
atf.disconnect()


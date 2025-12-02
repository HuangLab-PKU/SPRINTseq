from msl.loadlib import Server32
class ATFServer(Server32):
    def __init__(self,host,port,**kwargs):
        super().__init__('./atf_lib_dll.dll','cdll',host,port)
    def connect(self,com_port):
        ans = self.lib.ATF_OpenConnection('\\\\.\\'+com_port)
        return ans
    def disconnect(self):
        self.lib.ATF_CloseConnection()
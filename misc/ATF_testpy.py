from ctypes import *
atf_dll = windll.LoadLibrary('./atf_lib_dll.dll')
status = atf_dll.atf_OpenConnection(r'\\.\COM9',9600)
print(status)
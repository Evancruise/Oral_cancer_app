from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

print('gpu available?')
print(get_available_gpus())
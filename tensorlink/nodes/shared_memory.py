import json
import pickle
from copy import deepcopy
from multiprocessing import shared_memory


def store_in_shared_memory(_object, encoding=None):
    if encoding == "pickle":
        object_bytes = pickle.dumps(_object)
    elif encoding == "json":
        object_bytes = json.dumps(_object).encode()
    elif encoding is None:
        object_bytes = _object
    else:
        raise ValueError(f"Unsupported encoding: {encoding}")

    size = len(object_bytes)
    shm = shared_memory.SharedMemory(create=True, size=size)

    view = memoryview(shm.buf)
    view[:size] = object_bytes
    view.release()

    shm.close()

    return size, shm.name


def get_from_shared_memory(size, name, encoding=None):
    shm = shared_memory.SharedMemory(name=name)
    buffer = shm.buf[:size]

    if encoding == "pickle":
        _object = pickle.loads(buffer.tobytes())
    elif encoding == "json":
        _object = json.loads(buffer.tobytes()).encode()
    elif encoding is None:
        _object = buffer.tobytes()
    else:
        raise ValueError(f"Unsupported encoding: {encoding}")

    copied_object = deepcopy(_object)
    del buffer
    del _object
    shm.close()
    shm.unlink()
    return copied_object

from numba import cuda


@cuda.jit(device=True)
def a_device_function(a, b):
    return a + b


a_device_function(1, 3)

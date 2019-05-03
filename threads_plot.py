import numpy as np
import matplotlib.pyplot as plt

FILE_DIRS = ["openmp.txt"]

if __name__ == '__main__':
    # threads and time for size = 50000
    threads = [1,2,4,6,12,18,24,32]
    times = [11.3538,6.45065,4.0833,4.08696,4.03512,3.8876,3.5622,3.79546]
    plt.figure()
    plt.plot(threads, times, marker='o')

    my_x_ticks = np.arange(0, 34, 2)
    my_y_ticks = np.arange(0, 13, 1)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)

    plt.xlabel('number of threads')
    plt.ylabel('time/s')
    plt.grid(True)
    plt.show()
    

    # scale based on size = 10000, thread = 2
    scales = [1,2,3,6,9,12,16]
    times = [14.5471, 27.4374, 30.7889, 52.0214, 74.2369, 95.5548, 116.572]
    plt.figure()
    plt.plot(scales, times, marker='o')

    my_x_ticks = np.arange(0, 18, 1)
    my_y_ticks = np.arange(10, 120, 10)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)

    plt.xlabel('scale based on size=10000, #threads=2')
    plt.ylabel('time/s')
    plt.grid(True)

    plt.show()
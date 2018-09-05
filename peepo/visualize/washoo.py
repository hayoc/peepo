import random
from threading import Thread
from multiprocessing import Process


size = 10000000   # Number of random numbers to add to list
threads = 2 # Number of threads to create
my_list = []
for i in range(0,threads):
    my_list.append([])
def func(count, mylist):
    for i in range(count):
        mylist.append(random.random())
def multithreaded():
    jobs = []
    for i in range(0, threads):
        thread = Thread(target=func,args=(size,my_list[i]))
        jobs.append(thread)
    # Start the threads
    for j in jobs:
        j.start()
    # Ensure all of the threads have finished
    for j in jobs:
        j.join()

def simple():
    for i in range(0, threads):
        func(size,my_list[i])

def multiprocessed():
    processes = []
    for i in range(0, threads):
        p = Process(target=func,args=(size,my_list[i]))
        processes.append(p)
    # Start the processes
    for p in processes:
        p.start()
    # Ensure all processes have finished execution
    for p in processes:
        p.join()
if __name__ == "__main__":
    #multithreaded()
    #simple()
    multiprocessed()
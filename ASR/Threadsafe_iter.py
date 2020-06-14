import threading
'''
    A generic iterator and generator that takes any iterator and wrap it to make it thread safe.
    This method was introducted by Anand Chitipothu in http://anandology.com/blog/using-iterators-and-generators/
    but was not compatible with python 3. This modified version is now compatible and works both in python 2.8 and 3.0 
'''
class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()

def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g



'''
    Usage Examples. Here's how to use @threadsafe_generator to make any generator thread safe:
'''
@threadsafe_generator
def count():
    i = 0
    while True:
        i += 1
        yield i

'''
    This is a simple regular iterator, not thread safe.
'''
class Counter:
    def __init__(self):
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.i += 1
        return self.i


'''
    Running both examples in multithread example will show this actually works.
    
    Note: we compare our thread safe generator against a non thread safe iterator since a non thread safe generator
    in a multithreaded envrionment just crashes, yieldsing a  'ValueError: generator already executing' error.
    You can try it yourself by removing the @threadsafe_generator decorator from count().
    
'''

def loop(func, n):
    """Runs the given function n times in a loop.
    """
    for i in range(n):
        func()

def run(f, repeats=1000, nthreads=10):
    """Starts multiple threads to execute the given function multiple
    times in each thread.
    """
    # create threads
    threads = [threading.Thread(target=loop, args=(f, repeats))
               for i in range(nthreads)]

    # start threads
    for t in threads:
        t.start()

    # wait for threads to finish
    for t in threads:
        t.join()

def main():
    c1 = count()
    c2 = Counter()

    # call c1.next 100K times in 2 different threads
    run(c1.__next__, repeats=100000, nthreads=2)
    print ("c1", c1.__next__())

    # call c2.next 100K times in 2 different threads
    run(c2.__next__, repeats=100000, nthreads=2)
    print ("c2", c2.__next__())

if __name__ == "__main__":
    main()
def fibs():
    a,b = 0,1
    yield a
    yield b
    while True:
        a,b = b,a+b
        yield b

def nearest_fib(n):
    ''' If n is a Fibonacci number return True and n
        Otherwise, return False and the nearest Fibonacci number
    '''
    for fib in fibs():
        if fib == n:
            return n
        elif fib < n:
            prev = fib
        else:
            # Is n closest to prev or to fib?
            if n - prev < fib - n:
                return prev
            else:
                return fib

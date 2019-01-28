import functools #avoids confusion with the attributes of the decorated functions

def type_check(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except TypeError:
            print('Type Error:')
            print('The {} method of the {} class has one or more arguments with a wrong type.'
                  .format(func.__name__, type(args[0]).__name__))
            quit()
    return wrapper

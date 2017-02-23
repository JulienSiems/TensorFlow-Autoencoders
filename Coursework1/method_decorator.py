def method_decorator(method):
    def hahah(city_instance, input):
        if city_instance.name == "SFO":
            print("Its a cool place to live in.")
        else:
            method(city_instance, input)
    return hahah

class City(object):
    def __init__(self, name):
        self.name = name

    @method_decorator
    def print_test(self, input):
        print(self.name)
        print(input)

p1 = City("hey")

p1.print_test('hey')

@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator
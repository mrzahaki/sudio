"""
 W.T.A
 SUDIO (https://github.com/MrZahaki/sudio)
 The Audio Processing Platform
 Mail: mrzahaki@gmail.com
 Software license: "Apache License 2.0". See https://choosealicense.com/licenses/apache-2.0/
"""


class Register:
    def __init__(self):
        self.objects = []

    def parent(self, input_class):
        for method in self.objects:
            if type(method) is tuple:
                method = method[0]
                setattr(input_class, method.__name__, staticmethod(method))
            else:
                setattr(input_class, method.__name__, method)

        return input_class

    def add(self, obj):
        self.objects.append(obj)
        return obj

    def add_static(self, obj):
        self.objects.append((obj, ))
        return obj

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


# ___________________________________________________________def your objects here
class Members:
    sudio = Register()
    master = Register()
    process_slave = Register()

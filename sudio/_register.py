# the name of allah


class Register:
    def __init__(self):
        self.objects = []

    def parent(self, input_class):
        for method in self.objects:
            setattr(input_class, method.__name__, method)
        return input_class

    def add(self, obj):
        self.objects.append(obj)
        return obj


# ___________________________________________________________def your objects here
class Members:
    sudio = Register()

class Name:
    """
    Descriptor class for accessing and modifying the 'name' attribute of an object.

    This class is intended to be used as a descriptor for the 'name' attribute of an object.
    It allows getting and setting the 'name' attribute through the __get__ and __set__ methods.
    """

    def __get__(self, instance, owner):
        """Get the 'name' attribute of the associated object."""
        return instance._rec.name

    def __set__(self, instance, value):
        """Set the 'name' attribute of the associated object."""
        instance._rec.name = value

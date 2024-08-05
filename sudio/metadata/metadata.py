class AudioMetadata:
    def __init__(self, name, **kwargs):
        self.name = name
        self.__dict__.update(kwargs)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def keys(self):
        return [attr for attr in self.__dict__ if not attr.startswith('_')]

    def copy(self):
        return AudioMetadata(**{k: getattr(self, k) for k in self.keys()})

    def get_data(self):
        return self
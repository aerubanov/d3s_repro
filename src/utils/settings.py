class Settings:
    """ Training settings, e.g. the paths to datasets and networks."""
    def __init__(self):
        self.set_default()

    def set_default(self):
        self.use_gpu = True
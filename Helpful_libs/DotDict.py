class dotDict(dict):
    """
    dot notation to access keys of a dictionary

    Note
    ----
    You can only access string keys using this method
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__




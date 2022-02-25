"""
All things related to parsing ConfigParser arguments into proper datatypes.
"""
__author__ = 'ryanquinnnelson'


def to_int_list(s):
    """
    Build an integer list from a comma-separated string of integers.
    Args:
        s (str): comma-separated string of integers
    Returns:List
    """

    return [int(a) for a in s.strip().split(',')]


def to_string_list(s):
    """
    Build a list of strings from a comma-separated string.
    Args:
        s (str): comma-separated string of strings
    Returns:List
    """
    l1 = s.strip().split(',')
    return [each.strip() for each in l1]  # strip off newline characters when using formatted config file


def to_int_dict(s):
    """
    Build a dictionary where each value is an integer, given a comma-separated string of key=value pairs. If a value
    cannot be converted to an integer, leaves the value as a string.
    Args:
        s (str): comma-separated string of key=value pairs (i.e. key1=1,key2=2)
    Returns:Dict
    """

    d = dict()

    pairs = s.split(',')
    for p in pairs:
        key, val = p.strip().split('=')

        # try converting the value to an int
        try:
            val = int(val)
            continue # skip additional attempts to parse the type
        except ValueError:
            pass  # leave as string

        d[key] = val

    return d


def to_float_dict(s):
    """
    Build a dictionary where each value is a float, given a comma-separated string of key=value pairs. If a value
    cannot be converted to a float, leaves the value as a string.
    Args:
        s (str): comma-separated string of key=value pairs (i.e. key1=1,key2=2)
    Returns:Dict
    """

    d = dict()

    pairs = s.split(',')
    for p in pairs:
        key, val = p.strip().split('=')

        # try converting the value to a float
        try:
            val = float(val)
            continue # skip additional attempts to parse the type
        except ValueError:
            pass  # leave as string

        d[key] = val

    return d


def to_mixed_dict(s):
    """
    Build a dictionary where strings are converted to integers, floats, and boolean values (in that order) if they
    can be. Once a value is converted into one type, no additional conversions are attempted for that value.

    Args:
        s (str): comma-separated string of key=value pairs (i.e. key1=1,key2=2)

    Returns: Dict

    """
    d = dict()

    pairs = s.split(',')
    for p in pairs:
        key, val = p.strip().split('=')

        # try converting the value to an int
        try:
            val = int(val)
            d[key] = val
            continue  # skip additional attempts to parse the type
        except ValueError:
            pass  # leave as string

        # try converting the value to a float
        try:
            val = float(val)
            d[key] = val
            continue  # skip additional attempts to parse the type
        except ValueError:
            pass  # leave as string

        # try converting the value to a boolean
        try:
            if val == 'True':
                d[key] = True
                continue  # skip additional attempts to parse the type
            elif val == 'False':
                d[key] = False
                continue  # skip additional attempts to parse the type
            else:
                raise ValueError
        except ValueError:
            pass  # leave as string

        d[key] = val
    return d


def convert_configs_to_correct_type(config):
    """
    Build a dictionary where strings are converted to integers, floats, and boolean values (in that order) if they
    can be. Once a value is converted into one type, no additional conversions are attempted for that value.

    Args:
        config (Dict): Dictionary of key-value pairs.

    Returns:Dict

    """
    corrected_config = {}
    for key in config.keys():
        val = config[key]

        # try converting the value to an int
        try:
            val = int(val)
            corrected_config[key] = val
            continue  # skip additional attempts to parse the type
        except ValueError:
            pass  # leave as string

        # try converting the value to a float
        try:
            val = float(val)
            corrected_config[key] = val
            continue  # skip additional attempts to parse the type
        except ValueError:
            pass  # leave as string

        # try converting the value to a boolean
        try:
            if val == 'True':
                corrected_config[key] = True
                continue  # skip additional attempts to parse the type
            elif val == 'False':
                corrected_config[key] = False
                continue  # skip additional attempts to parse the type
            else:
                raise ValueError
        except ValueError:
            pass  # leave as string

        corrected_config[key] = val
    return corrected_config

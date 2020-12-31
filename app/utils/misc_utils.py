import datetime
import time
import re


def check_email_format(email):
    return bool(re.match(r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)", email))


def color_print(text='', color=0):
    """Print colored text.
    Args:
        text(str): text to print.
        color(int):
            * 0       black
            * 1       red
            * 2       green
            * 3       yellow
            * 4       blue
            * 5       cyan (like light red)
            * 6       magenta (like light blue)
            * 7       white
        end(str): end string after colored text.
    Example
        >>> color_print('yellow', 3)
    """
    print('\033[1;3%dm ' % color),
    print(text),
    print('\033[0m')


def get_time_stamp(add_offset=0):
    """Get time_zone+0 unix time stamp (seconds)
    Args:
        add_offset(int): bias added to time stamp
    Returns:
        (str): time stamp seconds
    """
    ti = int(time.time())
    ti = ti + add_offset
    return str(ti)


def get_time_str(time_stamp=get_time_stamp(), fmt="%Y/%m/%d %H:%M:%S", timezone=8, year_length=4):
    """Get formatted time string.

    Args:
        time_stamp(str): linux time string (seconds).
        fmt(str): string format.
        timezone(int): time zone.
        year_length(int): 2 or 4.
    Returns:
        (str): formatted time string.
    Example:
        >>> get_time_str()
        >>> # 2020/01/01 13:30:00
    """
    if not time_stamp:
        return ''

    time_stamp = int(time_stamp)

    base_time = datetime.datetime.utcfromtimestamp(time_stamp)

    time_zone_time = base_time + datetime.timedelta(hours=timezone)
    format_time_str = time_zone_time.strftime(fmt)

    if year_length == 2:
        format_time_str = format_time_str[2:]
    return format_time_str


def get_time_stamp_by_format_str(time_str, fmt="%Y/%m/%d %H:%M:%S", timezone=8):
    """Get timestamp by formatted time string.
    Args:
        time_str(str): string in fmt format.
        fmt(str): format.
        timezone(int): time zone.
    Returns:
        (str): time stamp
    Example:
        >>> get_time_stamp_by_format_str('2020/01/01 15:30:00')
        >>> # 1577863800
    """
    time_0 = datetime.datetime.utcfromtimestamp(0)

    time_str_parse = datetime.datetime.strptime(time_str, fmt)
    time_str_parse = time_str_parse - datetime.timedelta(hours=timezone)

    days = (time_str_parse - time_0).days
    seconds = (time_str_parse - time_0).seconds
    return str(days * 3600 * 24 + seconds)
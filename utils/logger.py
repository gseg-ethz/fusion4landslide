# source: geotrans/engine/logger.py; f2s3; spt/src/utils/pylogger.py

import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
import coloredlogs
# ignore 'PIL' logging warnings
logging.getLogger('PIL').setLevel(logging.WARNING)

field_styles = dict(
    asctime=dict(color='green'),
    hostname=dict(color='magenta'),
    levelname=dict(color='green'),
    filename=dict(color='magenta'),
    name=dict(color='blue'),
    threadName=dict(color='green')
)

level_styles = dict(
    debug=dict(color='green'),
    info=dict(color='cyan'),
    warning=dict(color='yellow'),
    error=dict(color='red'),
    critical=dict(color='red')
)


def get_logger(log_save_path):
    """
    Create a logger instance
    :param log_save_path:
    :return:
    """

    logger = logging.getLogger()
    fmt = "[%(asctime)s][%(levelname).4s] %(message)s [%(filename)s:%(lineno)d] "
    # hide filename
    # fmt = "[%(asctime)s][%(levelname).4s] %(message)s"
    # set the color indicates in the terminal
    coloredlogs.install(
        level="DEBUG",
        fmt=fmt,
        level_styles=level_styles,
        field_styles=field_styles)

    if log_save_path is not None:
        file_handler = logging.FileHandler(log_save_path)
        formatter = logging.Formatter(fmt, datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

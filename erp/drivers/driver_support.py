
__all__ = ['get_logger']

import logging
import time

current_version = "0.0.1"

def get_logger(log_format='%(asctime)s | %(levelname)s | %(message)s',
               date_format='%Y-%m-%d %H:%M:%S',
               log_name='logger',
               log_file_info='./erp.log'):
    """ Get a logger

    :param log_format:
    :param date_format:
    :param log_name:
    :param log_file_info:
    :return:
    """
    log = logging.getLogger(log_name)
    log_formatter = logging.Formatter(fmt=log_format, datefmt=date_format)
    logging.Formatter.converter = time.gmtime
    
    # comment this to suppress console output
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    log.addHandler(stream_handler)
    
    # eMRP.log with pipeline log
    file_handler_info = logging.FileHandler(log_file_info, mode='a')
    file_handler_info.setFormatter(log_formatter)
    file_handler_info.setLevel(logging.DEBUG)
    log.addHandler(file_handler_info)
    
    log.setLevel(logging.INFO)
    return log

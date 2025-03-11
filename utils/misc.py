from os import getpid
from psutil import Process
from yaml import load as yaml_load
from json import load as json_load

def parse_json_config(filename):
    with open(filename, 'r') as config_file:
        config_data = json_load(config_file)
    return config_data

def parse_yaml_config(filename):
    with open(filename, 'r') as config_file:
        config_data = yaml_load(config_file)
    return config_data

def get_memory_usage_mb():
    process = Process(getpid())
    mem_bytes = process.memory_info().rss
    return mem_bytes / (1024 * 1024)
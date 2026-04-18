#!/usr/bin/env python
# -*-coding:utf-8 -*-
# Version:1.1.8
import hashlib
# time:2025.6.3
# from 10:/D:/tool.py

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import TextIO
from random import Random

SELF_PATH = os.getcwd()
LOGGER = None
RANDOM = None
SEED = -1


def get_logger(name="logger", log_lv="INFO") -> 'Logger':
    global LOGGER
    if LOGGER is None:
        LOGGER = Logger(name)
        if log_lv == "TRACE":
            LOGGER.print_level = LOGGER.TRACE
        elif log_lv == "DEBUG":
            LOGGER.print_level = LOGGER.DEBUG
        if log_lv == "INFO":
            LOGGER.print_level = LOGGER.INFO
        if log_lv == "WARN":
            LOGGER.print_level = LOGGER.WARN
        if log_lv == "NOTICE":
            LOGGER.print_level = LOGGER.NOTICE
        if log_lv == "WARNING":
            LOGGER.print_level = LOGGER.WARN
        if log_lv == "ERROR":
            LOGGER.print_level = LOGGER.ERROR
        if log_lv == "CRITICAL":
            LOGGER.print_level = LOGGER.CRITICAL
    return LOGGER


def hash_str(s):
    try:
        return int(s)
    except ValueError:
        h = hashlib.sha256(s.encode('utf-8')).hexdigest()
        return int(h[:4], 16)


def get_random(seed: int = -1, new: bool = False) -> Random:
    global RANDOM, SEED
    if RANDOM is None or new:
        if seed == -1:
            seed = int((time.time() * 1e10) % (1e7 + 7))
            seed = hash_str(seed)
        SEED = seed
        get_logger().info("random seed: {}".format(seed))
        RANDOM = Random(seed)
    return RANDOM


class Logger:
    TRACE = 1
    DEBUG = 5
    INFO = 10
    NOTICE = 30
    WARN = 40
    ERROR = 60
    CRITICAL = 100
    time_format = "%Y-%m-%d %H:%M:%S"
    log_root = "log"

    def __init__(self, name: str,
                 lv=10,
                 max_size=262144,
                 log_path=None,
                 log_flag=True
                 ):
        self.print_level = lv
        self.max_size = max_size
        self.name = name
        self.log_flag = log_flag
        self.log_path = os.path.join(SELF_PATH, self.log_root) \
            if log_path is None else log_path

        self.file_id = 0
        self.file = None

        self.start()

    def __del__(self):
        self.close()

    def get_time(self):
        return time.strftime(self.time_format, time.localtime())

    def __create_file(self):
        if not os.path.exists(self.log_root):
            os.makedirs(self.log_root)
        file_name = os.path.join(self.log_root,
                                 f"{self.name}_{self.file_id}.log")
        while 1:
            if not Path(file_name).exists():
                open(file_name, 'x').close()
                break
            else:
                if os.path.getsize(file_name) > self.max_size:
                    self.file_id += 1
                    file_name = os.path.join(self.log_root, "{}_{}.log"
                                             .format(self.name, self.file_id))
                    continue
                else:
                    break
        self.file = open(file_name, 'a', encoding='utf-8')

    def __log(self, log_type, msg, log_lv, end="\n"):
        if self.print_level > log_lv:
            return
        s = f"<{self.get_time()}>" + f"[{log_type}]:" + f'{msg}{end}'
        print(s, end="", flush=True, file=sys.stderr)
        self.file.write(s)
        self.file.flush()

        if (self.max_size != -1 and
                os.path.getsize(self.file.name) > self.max_size):
            self.__create_file()

    def start(self):
        if self.file is None or self.file.closed:
            self.__create_file()
            if self.log_flag:
                self.__log("INFO", f"{self.name} log start", 4)

    def close(self):
        if not self.file.closed:
            if self.log_flag:
                self.__log("INFO", f"{self.name} log end\n", 4)
            self.file.close()

    def trace(self, msg, end="\n"):
        self.__log("TRACE", msg, log_lv=self.TRACE)

    def debug(self, msg, end="\n"):
        self.__log("DEBUG", msg, end=end, log_lv=self.DEBUG)

    def info(self, msg, end="\n"):
        self.__log("INFO", msg, end=end, log_lv=self.INFO)

    def warning(self, msg, end="\n"):
        self.__log("WARN", msg, end=end, log_lv=self.WARN)

    def warn(self, msg, end="\n"):
        self.__log("WARN", msg, end=end, log_lv=self.WARN)

    def notice(self, msg, end="\n"):
        self.__log("NOTICE", msg, end=end, log_lv=self.NOTICE)

    def error(self, msg, end="\n"):
        self.__log("ERROR", msg, end=end, log_lv=self.ERROR)

    def critical(self, msg, end="\n"):
        self.__log("CRITICAL", msg, end=end, log_lv=self.CRITICAL)


class GetData:
    def __init__(self, data_name, encoding="utf-8", data_path="data"):
        self.encoding = encoding
        self.data_name = data_name
        self.data_path = data_path
        self.io = None
        self.data = json.load(self.get_io())

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __load_io(self, mod="r"):
        return open(os.path.join(SELF_PATH, self.data_path, self.data_name + ".json"),
                    mod, encoding=self.encoding)

    def get_io(self, mod="r") -> TextIO:
        if self.io is None:
            self.io = self.__load_io(mod)
        if self.io is not None and self.io.mode != mod:
            self.io.close()
            self.io = self.__load_io(mod)
        return self.io

    def reload_data(self):
        self.io.close()
        self.io = None
        self.data = json.load(self.get_io("r"))

    def update_data(self):
        json.dump(self.get_io("w"), self.data, ensure_ascii=False, indent=4)
        self.io.close()
        self.io = None

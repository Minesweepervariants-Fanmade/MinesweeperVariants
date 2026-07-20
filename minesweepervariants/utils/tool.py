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
from typing import TextIO, Iterable
from random import Random

from minesweepervariants.config.config import DEFAULT_CONFIG

SELF_PATH = os.getcwd()
LOGGER = None
RANDOM = None
SEED = -1


def get_logger(name="logger", log_lv="INFO") -> 'Logger':
    global LOGGER
    log_lv = log_lv.upper() if log_lv else log_lv
    if LOGGER is None:
        LOGGER = Logger(name, log_path=DEFAULT_CONFIG.get("log_path", None))
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
        self.file_name = str(DEFAULT_CONFIG.get("log_file_name", "")).strip()
        self.use_file = bool(self.file_name)

        self.file_id = 0
        self.file = None

        self.start()

    def __del__(self):
        self.close()

    def get_time(self):
        return time.strftime(self.time_format, time.localtime())

    def __create_file(self):
        if not self.use_file:
            self.file = sys.stdout
            return
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        file_path = os.path.join(self.log_path, f"{self.file_name}.log")
        while 1:
            if not Path(file_path).exists():
                open(file_path, 'x').close()
                break
            else:
                if os.path.getsize(file_path) > self.max_size:
                    self.file_id += 1
                    file_path = os.path.join(self.log_path, f"{self.file_name}_{self.file_id}.log")
                    continue
                else:
                    break
        self.file = [open(file_path, 'a', encoding='utf-8'), sys.stdout]

    def __log(self, log_type, msg, log_lv, *args, **kwargs):
        if self.print_level > log_lv:
            return
        end = kwargs.pop('end', "\n")
        flush = kwargs.pop('flush', True)
        file_arg = kwargs.pop('file', self.file)
        s = f"<{self.get_time()}>" + f"[{log_type}]:" + f'{msg}{end}'
        if isinstance(file_arg, (list, tuple)):
            for file_obj in file_arg:
                print(s, *args, **kwargs, end="", flush=flush, file=file_obj)
        else:
            print(s, *args, **kwargs, end="", flush=flush, file=file_arg)

        # if (self.use_file and file_obj is not None and self.max_size != -1 and
        #         os.path.getsize(file_obj.name) > self.max_size):
        #     self.__create_file()

    def start(self):
        if self.file is None or self.file.closed:
            self.__create_file()
            if self.log_flag:
                self.__log("INFO", f"{self.name} log start", 4)

    def close(self):
        try:
            if self.file is not None and not self.file.closed:
                if self.log_flag:
                    self.__log("INFO", f"{self.name} log end\n", 4)
                if self.use_file:
                    self.file.close()
        except:
            pass

    def trace(self, msg, *args, **kwargs):
        self.__log("TRACE", msg=msg, log_lv=self.TRACE, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self.__log("DEBUG", msg=msg, log_lv=self.DEBUG, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.__log("INFO", msg=msg, log_lv=self.INFO, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.__log("WARN", msg=msg, log_lv=self.WARN, *args, **kwargs)

    def warn(self, msg, *args, **kwargs):
        self.__log("WARN", msg=msg, log_lv=self.WARN, *args, **kwargs)

    def notice(self, msg, *args, **kwargs):
        self.__log("NOTICE", msg=msg, log_lv=self.NOTICE, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.__log("ERROR", msg=msg, log_lv=self.ERROR, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self.__log("CRITICAL", msg=msg, log_lv=self.CRITICAL, *args, **kwargs)


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

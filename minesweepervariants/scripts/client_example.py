#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
示例客户端：连接 `TerminalEmulator`，发送启动参数并支持与子进程的交互。
用法示例：
  python minesweepervariants/scripts/client_example.py --host 127.0.0.1 --port 31408 -- -s 5 -c JB -t 10
注意： `--` 后面的所有内容会按空格拼接并传给服务端作为参数行。
"""

import argparse
import socket
import os
import select
try:
    import termios
    import tty
except Exception:
    termios = None
    tty = None
try:
    import msvcrt
except Exception:
    msvcrt = None
import threading
import sys


def recv_thread(sock):
    try:
        while True:
            try:
                data = sock.recv(4096)
            except socket.timeout:
                # 如果使用了超时，继续等待数据
                continue
            if not data:
                break
            # 支持 debug 输出：如果外部设置了 sock._debug_bytes，则额外打印接收到的原始字节 repr
            if getattr(sock, '_debug_bytes', False):
                print(f"[DEBUG recv bytes] {repr(data)}", file=sys.stderr)
            sys.stdout.write(data.decode('utf-8', errors='replace'))
            sys.stdout.flush()
    except Exception as e:
        print(f"接收数据时出错: {e}", file=sys.stderr)


def stdin_thread(sock, raw=False):
    try:
        if raw:
            # 原始模式：逐字符读取并立即发送（跨平台）
            if os.name == 'nt' and msvcrt:
                try:
                    while True:
                        ch = msvcrt.getwch()
                        if not ch:
                            break
                        try:
                            sock.sendall(ch.encode('utf-8'))
                        except Exception as e:
                            print(f"发送到服务端时出错: {e}", file=sys.stderr)
                            break
                except KeyboardInterrupt:
                    return
            else:
                # Unix-like: 切换到 raw 模式
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                try:
                    tty.setraw(fd)
                    while True:
                        ch = sys.stdin.read(1)
                        if not ch:
                            break
                        try:
                            sock.sendall(ch.encode('utf-8'))
                        except Exception as e:
                            print(f"发送到服务端时出错: {e}", file=sys.stderr)
                            break
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        else:
            while True:
                line = sys.stdin.readline()
                if not line:
                    break
                try:
                    sock.sendall(line.encode('utf-8'))
                except Exception as e:
                    print(f"发送到服务端时出错: {e}", file=sys.stderr)
                    break
    except Exception as e:
        print(f"读取控制台输入时出错: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description='示例客户端 for TerminalEmulator')
    parser.add_argument('--host', default='127.0.0.1', help='服务器地址 (默认: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=31408, help='服务器端口 (默认: 31408)')
    parser.add_argument('args', nargs=argparse.REMAINDER, help='要传给 minesweepervariants 的参数，使用 -- 分隔')
    parser.add_argument('--raw', action='store_true', help='启用原始输入模式，按键会立即发送（无需回车）')
    parser.add_argument('--debug', action='store_true', help='打印接收的原始字节 repr，便于调试输出问题')
    ns = parser.parse_args()

    host = ns.host
    port = ns.port
    send_args = ' '.join(ns.args).strip()

    try:
        with socket.create_connection((host, port), timeout=10) as s:
            # 连接建立后去掉超时，避免 recv 抛出 socket.timeout
            s.settimeout(None)
            # 读取欢迎字节（脚本发送 4 字节 '14mv'）
            try:
                welcome = s.recv(4)
                if welcome:
                    print(welcome.decode('utf-8', errors='replace'))
            except Exception:
                pass

            # 发送参数行（如果有），并以换行结束
            if send_args:
                s.sendall((send_args + "\n").encode('utf-8'))

            # 启动接收线程和 stdin 转发线程
            # 如果需要 debug 原始字节，设置 socket 属性以便 recv_thread 读取
            if ns.debug:
                setattr(s, '_debug_bytes', True)
            rt = threading.Thread(target=recv_thread, args=(s,), daemon=True)
            st = threading.Thread(target=stdin_thread, args=(s, ns.raw), daemon=True)
            rt.start()
            st.start()

            # 等待接收线程结束（当服务器关闭连接时结束）
            rt.join()
    except ConnectionRefusedError:
        print(f"无法连接到 {host}:{port}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"客户端错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

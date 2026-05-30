#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# @Time    : 2025/07/26 13:29
# @Author  : Wu_RH
# @FileName: api.py
import signal
import socket
import sys
import threading
import subprocess
import os
import time
from datetime import datetime
from typing import Optional

import select
from queue import Queue, Empty


def kill_process_tree(process):
    """终止进程及其所有子进程（跨平台）"""
    if process.poll() is None:  # 进程仍在运行
        try:
            if os.name == 'nt':  # Windows 系统
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True
                )
            else:  # Unix/Linux 系统
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except (OSError, subprocess.CalledProcessError):
            process.kill()  # 兜底方案：强制终止
        process.wait()  # 确保资源回收


class TerminalEmulator:
    def __init__(
        self, _port=5000,
        host='0.0.0.0',
        front_arg=None
    ):
        if front_arg is None:
            front_arg = [
                "uv", "run", "python",
                "-m", "minesweepervariants"
            ]
        self.port = _port
        self.host = host
        self.front_arg = front_arg
        self.server_socket: Optional[socket.socket] = None
        self.running = False
        self.client_lock = threading.Lock()
        self.clients = {}
        self.output_queues = {}
        self.process_count = 0  # 添加进程计数器

    def start_server(self):
        """启动终端服务器"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.running = True

        def get_all_ips():
            try:
                import netifaces
            except ImportError:
                print("请安装 netifaces 模块以获取本机IP地址: pip install netifaces")
                return []
            ips = []
            for iface in netifaces.interfaces():
                addrs = netifaces.ifaddresses(iface)
                # IPv4地址
                if netifaces.AF_INET in addrs:
                    for addr in addrs[netifaces.AF_INET]:
                        ips.append(addr['addr'])
                # IPv6地址（过滤掉链路本地地址和临时地址，根据需要保留）
                # if netifaces.AF_INET6 in addrs:
                #     for addr in addrs[netifaces.AF_INET6]:
                #         ipv6 = addr['addr']
                #         # 去掉IPv6地址末尾的百分号区域ID，例如 'fe80::1%eth0' -> 'fe80::1'
                #         if '%' in ipv6:
                #             ipv6 = ipv6.split('%')[0]
                #         ips.append(ipv6)
            return ips

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 终端服务器启动在 {self.host}:{self.port}")
        if self.host == "0.0.0.0":
            all_ips = get_all_ips()
            print("本机所有IP地址：")
            for ip in all_ips:
                print(f"{ip}:{self.port}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 等待连接，使用 {self.front_arg} 执行命令minesweepervariants..")

        # 启动主接收线程
        threading.Thread(target=self._accept_clients, daemon=True).start()

    def stop_server(self):
        """停止服务器"""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 服务器已停止")

    def _accept_clients(self):
        """接受客户端连接"""
        while self.running:
            try:
                client_socket, addr = self.server_socket.accept()
                # print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 新连接来自: {addr[0]}:{addr[1]}")

                # 为新客户端创建输出队列
                output_queue = Queue()

                with self.client_lock:
                    self.clients[client_socket] = {
                        'address': addr,
                        'process': None,
                        'active': True,
                        'output_queue': output_queue
                    }

                # 发送初始字节 "14mv"
                client_socket.sendall(b"14mv")

                # 启动客户端处理线程
                threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, output_queue),
                    daemon=True
                ).start()

            except OSError:
                if self.running:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 接受连接时出错")
                break

    def _handle_client(self, client_socket: socket.socket, output_queue):
        """处理单个客户端连接"""
        process = None
        try:
            # 等待接收客户端参数
            data = client_socket.recv(0xffffffff)
            if not data:
                # print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 客户端未发送参数，断开连接")
                return

            # 解析参数
            args = data.decode('utf-8').strip().split()
            args_print = f"{[arg if len(arg) < 10 else f'{arg[:3]}...{arg[-3:]}' for arg in args]}"
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 接收到参数: {args_print}")

            if os.name == 'nt':
                os.system('')  # 关键！激活 ANSI 和实时输出

            # 创建子进程执行命令
            process = subprocess.Popen(
                self.front_arg + args,
                cwd=os.getcwd(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                # 确保 Python 子进程使用无缓冲输出，以便提示符和交互及时可见
                env=dict(os.environ, PYTHONUNBUFFERED='1'),
                encoding='utf-8',  # 指定编码
                errors='replace',  # 替换无法解码的字符
                universal_newlines=True,   # 自动处理换行符
                bufsize=1,
                shell=False
            )
            with self.client_lock:
                self.process_count += 1  # 增加进程计数

            # 更新客户端信息
            with self.client_lock:
                self.clients[client_socket]['process'] = process

            # 启动输出捕获线程
            threading.Thread(
                target=self._capture_output,
                args=(process, output_queue, args, client_socket),
                daemon=True
            ).start()

            # 主循环：处理客户端请求
            while True:
                # 检查进程是否结束
                if process.poll() is not None:
                    exit_code = process.returncode
                    # 发送剩余输出
                    self._send_output(client_socket, output_queue)
                    # 发送退出码
                    client_socket.sendall(f"\nExit Code: {exit_code}".encode('utf-8'))
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 进程结束，退出码: {exit_code}")
                    break

                # 每隔0.1秒一次更新
                time.sleep(0.1)

                # 检查是否有来自客户端的输入
                readable, _, _ = select.select([client_socket], [], [], 0.1)
                if client_socket in readable:
                    data = client_socket.recv(1024)
                    if not data:
                        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 客户端断开连接")
                        break
                    # 将客户端输入转发到子进程的 stdin，以支持交互式游戏
                    try:
                        if process and process.poll() is None and process.stdin:
                            # 以 utf-8 解码并写入子进程 stdin
                            text = data.decode('utf-8', errors='replace')
                            process.stdin.write(text)
                            process.stdin.flush()
                    except Exception as e:
                        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 转发客户端输入到子进程时出错: {str(e)}")
                self._send_output(client_socket, output_queue)

        except (ConnectionResetError, BrokenPipeError):
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 客户端连接意外断开")
        except Exception as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 处理客户端时出错: {str(e)}")
        finally:
            # 清理资源 - 确保进程终止
            if process:
                try:
                    # 先尝试关闭 stdin，允许进程自然退出
                    try:
                        if process.stdin:
                            process.stdin.close()
                    except Exception:
                        pass
                    kill_process_tree(process)
                    with self.client_lock:
                        self.process_count -= 1  # 减少进程计数
                except Exception as e:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 终止进程时出错: {str(e)}")

            with self.client_lock:
                if client_socket in self.clients:
                    del self.clients[client_socket]

            try:
                client_socket.close()
            except:
                pass
            # print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 客户端连接已关闭")

    @staticmethod
    def _capture_output(process: subprocess.Popen, output_queue, args, client_socket):
        """捕获子进程输出"""
        output_queue.put(f"PID:[{process.pid}]\n")
        try:
            # 按字符读取 stdout，避免等待换行，从而让提示符等无换行输出也能及时到达客户端
            while True:
                ch = process.stdout.read(1)
                if ch == '' or ch is None:
                    # 无可读内容
                    if process.poll() is not None:
                        break
                    # 进程未退出但当前无数据，短暂休眠后继续
                    time.sleep(0.01)
                    continue
                # 读取到字符，加入队列（字符或小片段）
                output_queue.put(ch)
                # 立即尝试把队列里的内容发送给对应客户端，实现“有字符立即转发”
                try:
                    if client_socket.fileno() != -1:
                        out_parts = []
                        while True:
                            try:
                                out_parts.append(output_queue.get_nowait())
                            except Empty:
                                break
                        if out_parts:
                            try:
                                client_socket.sendall(''.join(out_parts).encode('utf-8'))
                            except Exception:
                                # 发送失败则把内容放回队列供后续发送尝试
                                for p in out_parts:
                                    output_queue.put(p)
                except Exception:
                    pass
            stderr = process.stderr.read()
            if stderr:
                output_queue.put("\n[STDERR]:\n" + stderr + "\n:[STDERR]")
            else:
                output_queue.put("\n[STDERR EMPTY]\n")
        except ValueError as e:
            # 当管道关闭时可能发生
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 管道关闭: {str(e)}")
        except Exception as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 捕获输出时出错: {str(e)}")
        args_print = f"{[arg if len(arg) < 10 else f'{arg[:3]}...{arg[-3:]}' for arg in args]}"
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {args_print}: 退出")

    @staticmethod
    def _send_output(client_socket, output_queue):
        """发送输出队列中的所有内容给客户端"""
        if client_socket.fileno() == -1:
            return  # 套接字已关闭

        output_lines = []
        while True:
            try:
                line = output_queue.get_nowait()
                output_lines.append(line)
            except Empty:
                break

        if output_lines:
            output = ''.join(output_lines)
            try:
                client_socket.sendall(output.encode('utf-8'))
            except (ConnectionResetError, BrokenPipeError):
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 发送输出时连接已断开")
            except Exception as e:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 发送输出时出错: {str(e)}")


if __name__ == "__main__":
    # 创建并启动终端仿真器
    import argparse

    parser = argparse.ArgumentParser(description='终端仿真器')
    parser.add_argument('-H', '--host', default='0.0.0.0', help='监听的主机地址 (默认: 0.0.0.0)')
    parser.add_argument('-p', '--port', type=int, default=31408, help='监听的端口号 (默认: 31408)')
    args = parser.parse_args()

    emulator = TerminalEmulator(
        _port=args.port,
        host=args.host,
    )
    print_rate = 60
    try:
        emulator.start_server()
        # 保持主线程运行
        last_print_time = time.time()
        while True:
            current_time = time.time()
            if current_time - last_print_time >= print_rate:
                # 打印一次进程数量
                if emulator.process_count:
                    with emulator.client_lock:
                        active_clients = len(emulator.clients)
                        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 当前运行中的进程数: {emulator.process_count}, 活跃客户端数: {active_clients}")
                    last_print_time = current_time
                    print_rate <<= 1
                    if print_rate > 3600:
                        print_rate = 3600
            if not emulator.process_count:
                print_rate = 60
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 接收到中断信号，停止服务器minesweepervariants..")
    finally:
        emulator.stop_server()

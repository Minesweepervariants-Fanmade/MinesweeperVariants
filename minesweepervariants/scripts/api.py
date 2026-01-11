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
    def __init__(self, _port=5000, host='0.0.0.0', bat_file='run.bat'):
        self.port = _port
        self.host = host
        self.bat_file = bat_file
        self.server_socket = None
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

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 终端服务器启动在 {self.host}:{self.port}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 等待连接，使用 {self.bat_file} 执行命令minesweepervariants..")

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
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 新连接来自: {addr[0]}:{addr[1]}")

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

    def _handle_client(self, client_socket, output_queue):
        """处理单个客户端连接"""
        process = None
        with self.client_lock:
            self.process_count += 1  # 增加进程计数
        try:
            # 等待接收客户端参数
            data = client_socket.recv(1024)
            if not data:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 客户端未发送参数，断开连接")
                return

            # 解析参数
            args = data.decode('utf-8').strip().split()
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 接收到参数: {args}")

            if os.name == 'nt':
                os.system('')  # 关键！激活 ANSI 和实时输出

            # 创建子进程执行命令
            process = subprocess.Popen(
                [self.bat_file] + args,
                cwd=os.getcwd(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                encoding='gbk',  # 指定编码
                errors='replace',  # 替换无法解码的字符
                universal_newlines=True,   # 自动处理换行符
                bufsize=1,
                shell=False
            )

            # 更新客户端信息
            with self.client_lock:
                self.clients[client_socket]['process'] = process

            # 启动输出捕获线程
            threading.Thread(
                target=self._capture_output,
                args=(process, output_queue),
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

                self._send_output(client_socket, output_queue)

        except (ConnectionResetError, BrokenPipeError):
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 客户端连接意外断开")
        except Exception as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 处理客户端时出错: {str(e)}")
        finally:
            # 清理资源 - 确保进程终止
            if process:
                try:
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
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 客户端连接已关闭")

    @staticmethod
    def _capture_output(process, output_queue):
        """捕获子进程输出"""
        try:
            while True:
                line = os.read(process.stdout.fileno(), 1024)
                line = line.decode("gbk") if line else ''
                if not line and process.poll() is not None:
                    break
                if line:
                    # 将输出添加到队列
                    output_queue.put(line)
        except ValueError as e:
            # 当管道关闭时可能发生
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 管道关闭: {str(e)}")
        except Exception as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 捕获输出时出错: {str(e)}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 退出")

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
    port = sys.argv[1] if len(sys.argv) > 1 else 31408
    emulator = TerminalEmulator(
        _port=int(port),  # 监听端口
        host='0.0.0.0',  # 监听所有接口
        bat_file='run.bat'  # 要执行的批处理文件
    )

    try:
        emulator.start_server()
        # 保持主线程运行
        last_print_time = time.time()
        while True:
            current_time = time.time()
            # 每分钟打印一次进程数量
            if (
                current_time - last_print_time >= 60 and
                emulator.process_count
            ):  # 60秒 = 1分钟
                with emulator.client_lock:
                    active_clients = len(emulator.clients)
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 当前运行中的进程数: {emulator.process_count}, 活跃客户端数: {active_clients}")
                last_print_time = current_time
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 接收到中断信号，停止服务器minesweepervariants..")
    finally:
        emulator.stop_server()

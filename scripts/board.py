"""Shared board utilities for ZyboGPT scripts."""

import os
import shutil
import subprocess
import serial
import serial.tools.list_ports
import sys
import threading
import time


def find_serial_port():
    """Find the Zybo FTDI serial port (cross-platform).

    Returns the device path (/dev/ttyUSB1 on Linux, /dev/cu.usbserial-* on
    macOS, COMx on Windows) or None if no matching device is found.
    """
    # Prefer FTDI FT2232H (Zybo): VID 0x0403, PID 0x6010
    for p in serial.tools.list_ports.comports():
        if p.vid == 0x0403 and p.pid == 0x6010:
            return p.device
    # Fallback: any FTDI device
    for p in serial.tools.list_ports.comports():
        if p.vid == 0x0403:
            return p.device
    return None


def find_xsdb():
    """Find XSDB executable."""
    # 1. Environment variable
    env = os.environ.get('XSDB')
    if env:
        return env
    # 2. PATH
    path = shutil.which('xsdb')
    if path:
        return path
    # 3. Common install locations
    for base in [
        os.path.expanduser('~/tools/Xilinx/Vivado'),
        '/opt/Xilinx/Vivado',
        '/tools/Xilinx/Vivado',
        'C:/Xilinx/Vivado',
    ]:
        if os.path.isdir(base):
            for ver in sorted(os.listdir(base), reverse=True):
                candidate = os.path.join(base, ver, 'bin', 'xsdb')
                if os.path.isfile(candidate):
                    return candidate
    return None


XSDB = find_xsdb()
MAILBOX_BASE = 0xFFFF_F000
BAUD = 115200
DONE_MARKER = '__XSDB_DONE__'


# ── Input backends ──────────────────────────────────────────────


class UartBackend:
    """Send commands directly over UART serial (when RX works)."""

    def __init__(self, ser):
        self.ser = ser

    def send(self, cmd_str: str):
        self.ser.write((cmd_str + '\r\n').encode('ascii'))
        self.ser.flush()

    def send_stop(self):
        self.ser.write(b'\r\n')
        self.ser.flush()

    def close(self):
        pass


class XsdbBackend:
    """Persistent XSDB process that writes to the OCM mailbox over JTAG."""

    def __init__(self):
        self.proc = None
        self._reader_thread = None
        self._output_buf = ''
        self._lock = threading.Lock()
        self._done_event = threading.Event()

    def connect(self):
        if XSDB is None:
            raise FileNotFoundError(
                'XSDB not found. Set the XSDB environment variable or add '
                'it to PATH.')
        self.proc = subprocess.Popen(
            [XSDB],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

        self._exec('connect')
        self._exec('after 500')
        self._exec(
            'targets -set -nocase -filter '
            '{name =~ "*Cortex*#0" || name =~ "*ARM*#0"}'
        )
        # Stop any running bench from a previous session + clear stale mailbox
        self._exec(
            f'mwr 0x{MAILBOX_BASE + 4:08X} 0\n'
            f'mwr 0x{MAILBOX_BASE:08X} 1'
        )
        self._exec('after 200')
        self._exec(f'mwr 0x{MAILBOX_BASE:08X} 0')

    def _read_loop(self):
        try:
            while True:
                chunk = self.proc.stdout.read(1)
                if not chunk:
                    break
                ch = chunk.decode('ascii', errors='replace')
                with self._lock:
                    self._output_buf += ch
                    if DONE_MARKER in self._output_buf:
                        self._done_event.set()
                        self._output_buf = ''
        except Exception:
            pass

    def _exec(self, tcl_script: str, timeout=15):
        self._done_event.clear()
        with self._lock:
            self._output_buf = ''
        payload = tcl_script + f'\nputs {{{DONE_MARKER}}}\n'
        self.proc.stdin.write(payload.encode('ascii'))
        self.proc.stdin.flush()
        if not self._done_event.wait(timeout=timeout):
            raise TimeoutError(f'XSDB timed out on: {tcl_script[:60]}')

    def send(self, cmd_str: str):
        if self.proc is None or self.proc.poll() is not None:
            self.connect()

        cmd_bytes = cmd_str.encode('ascii')[:256]
        tcl_lines = []
        data_addr = MAILBOX_BASE + 8
        for i in range(0, len(cmd_bytes), 4):
            chunk = cmd_bytes[i:i + 4]
            val = 0
            for j, b in enumerate(chunk):
                val |= b << (j * 8)
            tcl_lines.append(f'mwr 0x{data_addr + i:08X} 0x{val:08X}')

        tcl_lines.append(f'mwr 0x{MAILBOX_BASE + 4:08X} {len(cmd_bytes)}')
        tcl_lines.append(f'mwr 0x{MAILBOX_BASE:08X} 1')
        self._exec('\n'.join(tcl_lines))

    def send_stop(self):
        """Set mailbox ready flag to stop firmware (minimal, fast).

        Uses a short timeout and raw write fallback so it never blocks
        the keypress handler for long.
        """
        if self.proc is None or self.proc.poll() is not None:
            return
        try:
            self._exec(
                f'mwr 0x{MAILBOX_BASE + 4:08X} 0\n'
                f'mwr 0x{MAILBOX_BASE:08X} 1',
                timeout=2,
            )
        except (TimeoutError, Exception):
            # _exec timed out (XSDB unresponsive) — fire-and-forget raw write
            try:
                raw = (
                    f'mwr 0x{MAILBOX_BASE + 4:08X} 0\n'
                    f'mwr 0x{MAILBOX_BASE:08X} 1\n'
                ).encode('ascii')
                self.proc.stdin.write(raw)
                self.proc.stdin.flush()
            except Exception:
                pass

    def close(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.write(b'exit\n')
                self.proc.stdin.flush()
                self.proc.wait(timeout=3)
            except Exception:
                self.proc.kill()


# ── Serial reader ───────────────────────────────────────────────


class SerialReader:
    """Background thread that streams UART output and detects the firmware prompt."""

    def __init__(self, port):
        self.port = port
        self.ser = None
        self.stop_event = threading.Event()
        self.prompt_event = threading.Event()
        self.thread = None
        self._buf = ''
        self._lock = threading.Lock()
        self.suppress_prompt = True
        self.silent = False
        self._write_lock = threading.Lock()

    def start(self):
        self.ser = serial.Serial(
            self.port, BAUD, timeout=0.1,
            xonxoff=False, rtscts=False, dsrdtr=False,
        )
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop_event.is_set():
            try:
                data = self.ser.read(1024)
            except Exception:
                break
            if not data:
                continue

            text = data.decode('ascii', errors='replace').replace('\r', '')

            with self._lock:
                self._buf += text
                if len(self._buf) > 1024:
                    self._buf = self._buf[-512:]
                if '> ' in self._buf:
                    self.prompt_event.set()

            if not self.silent:
                display = text
                if self.suppress_prompt:
                    display = display.replace('\n> ', '\n')
                    if display.endswith('> '):
                        display = display[:-2]
                    if display == '> ':
                        display = ''
                if display:
                    with self._write_lock:
                        sys.stdout.write(display)
                        sys.stdout.flush()

    def drain(self, seconds=0.5):
        time.sleep(seconds)

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2)
        if self.ser:
            self.ser.close()


# ── UART RX probe ──────────────────────────────────────────────


def probe_uart_rx(port):
    """Send CONFIG via UART and check if we get a response back."""
    try:
        ser = serial.Serial(
            port, BAUD, timeout=2,
            xonxoff=False, rtscts=False, dsrdtr=False,
        )
        ser.reset_input_buffer()
        ser.write(b'CONFIG\r\n')
        ser.flush()
        time.sleep(1.5)
        data = ser.read(4096)
        ser.close()
        return b'd_model=' in data
    except Exception:
        return False

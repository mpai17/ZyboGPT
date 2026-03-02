#!/usr/bin/env python3
"""Interactive console for ZyboGPT — ollama-style.

Auto-detects whether UART RX works. If it does, uses direct serial I/O
(instant). If not, falls back to a persistent XSDB connection that sends
commands via an OCM mailbox (fast — no process spawn per command).

Usage:
    python scripts/cmd.py                  # interactive mode
    python scripts/cmd.py CONFIG           # single command
    python scripts/cmd.py "ROMEO:"         # generate from prompt
"""
import sys
import time

from board import (
    find_serial_port, probe_uart_rx,
    UartBackend, XsdbBackend, SerialReader,
)

IS_WINDOWS = sys.platform == 'win32'
if not IS_WINDOWS:
    import select, termios, tty
else:
    import msvcrt


OUTPUT_TIMEOUT = 330  # 5 min bench + margin


# ── Main ────────────────────────────────────────────────────────


BOLD = '\033[1m'
DIM = '\033[2m'
RST = '\033[0m'
CYAN = '\033[36m'
GREEN = '\033[32m'
YELLOW = '\033[33m'


def interactive(port):
    print(f'{BOLD}{"=" * 50}{RST}')
    print(f'{BOLD}  ZyboGPT — Ternary LLM on Zybo Z7-10 FPGA{RST}')
    print(f'{BOLD}{"=" * 50}{RST}')
    print()

    print(f'{DIM}  Checking UART RX...{RST}', end='', flush=True)
    uart_rx_ok = probe_uart_rx(port)

    if uart_rx_ok:
        print(f'\r  Input: {GREEN}{BOLD}UART serial{RST} (direct)')
    else:
        print(f'\r  Input: {YELLOW}{BOLD}XSDB mailbox{RST} (UART RX not available)')

    reader = SerialReader(port)
    reader.start()
    reader.drain(0.3)

    if uart_rx_ok:
        backend = UartBackend(reader.ser)
    else:
        backend = XsdbBackend()
        try:
            backend.connect()
        except Exception as e:
            print(f'{DIM}  XSDB connect failed: {e}{RST}', file=sys.stderr)
            reader.stop()
            return

    print()
    print(f'  Type text to use as a generation prompt.')
    print(f'  Commands: {BOLD}CONFIG{RST}, {BOLD}BENCH{RST}')
    print(f'  {DIM}"quit" to exit{RST}')
    print()

    while True:
        try:
            cmd = input(f'{CYAN}{BOLD}>>> {RST}')
        except (EOFError, KeyboardInterrupt):
            break

        cmd = cmd.strip()
        if not cmd:
            continue
        if cmd.lower() in ('quit', 'exit'):
            break

        try:
            backend.send(cmd)
        except Exception as e:
            print(f'\n{DIM}[send error: {e}]{RST}', file=sys.stderr)
            if isinstance(backend, XsdbBackend):
                print(f'{DIM}[reconnecting...]{RST}', file=sys.stderr)
                try:
                    backend.close()
                    backend = XsdbBackend()
                    backend.connect()
                    backend.send(cmd)
                except Exception as e2:
                    print(f'{DIM}[reconnect failed: {e2}]{RST}', file=sys.stderr)
                    continue
            else:
                continue

        reader.prompt_event.clear()
        with reader._lock:
            reader._buf = ''
        _wait_for_response(reader, backend)
        print()

    print('\nBye!')
    backend.close()
    reader.stop()


def _wait_for_response(reader, backend):
    """Wait for firmware '> ' prompt. Any keypress sends a stop signal."""
    is_tty = sys.stdin.isatty()
    last_stop = 0.0  # rate-limit stop attempts
    old_settings = None

    if is_tty and not IS_WINDOWS:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)

    try:
        deadline = time.time() + OUTPUT_TIMEOUT
        while time.time() < deadline:
            if reader.prompt_event.is_set():
                reader.drain(0.2)
                return

            # Any keypress (or Ctrl+C) sends a stop signal; allow retries
            if is_tty:
                key_pressed = False
                if IS_WINDOWS:
                    if msvcrt.kbhit():
                        msvcrt.getch()
                        key_pressed = True
                else:
                    if select.select([sys.stdin], [], [], 0)[0]:
                        sys.stdin.read(1)
                        key_pressed = True
                if key_pressed:
                    now = time.time()
                    if now - last_stop > 1.0:  # rate-limit to 1 stop/sec
                        try:
                            backend.send_stop()
                        except Exception:
                            pass
                        last_stop = now

            time.sleep(0.05)

        if not reader.prompt_event.is_set():
            sys.stdout.write(f'\n{DIM}[timeout]{RST}\n')
            sys.stdout.flush()

    except KeyboardInterrupt:
        try:
            backend.send_stop()
        except Exception:
            pass
        deadline = time.time() + 15
        while time.time() < deadline:
            if reader.prompt_event.is_set():
                reader.drain(0.2)
                break
            time.sleep(0.05)

    finally:
        if is_tty and not IS_WINDOWS:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        reader.prompt_event.clear()
        with reader._lock:
            reader._buf = ''


def oneshot(port, cmd):
    uart_rx_ok = probe_uart_rx(port)

    reader = SerialReader(port)
    reader.suppress_prompt = False
    reader.start()
    reader.drain(0.3)

    if uart_rx_ok:
        backend = UartBackend(reader.ser)
    else:
        backend = XsdbBackend()
        backend.connect()

    backend.send(cmd)

    deadline = time.time() + OUTPUT_TIMEOUT
    while time.time() < deadline:
        if reader.prompt_event.is_set():
            reader.drain(0.2)
            break
        time.sleep(0.1)
    else:
        time.sleep(3)

    backend.close()
    reader.stop()
    print()


def main():
    port = find_serial_port()
    if port is None:
        print('Error: no Zybo serial port found (board not connected?)',
              file=sys.stderr)
        sys.exit(1)

    if len(sys.argv) > 1:
        oneshot(port, ' '.join(sys.argv[1:]))
    else:
        interactive(port)


if __name__ == '__main__':
    main()

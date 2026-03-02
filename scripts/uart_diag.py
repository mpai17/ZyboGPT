#!/usr/bin/env python3
"""UART RX diagnostic for ZyboGPT.

Tests bidirectional communication with explicit flow control settings.
"""
import serial
import time
import sys
import os

PORT = '/dev/ttyUSB1'
BAUD = 115200

def check_port_settings(ser):
    """Print current serial port settings."""
    print(f"Port: {ser.port}")
    print(f"Baudrate: {ser.baudrate}")
    print(f"Bytesize: {ser.bytesize}")
    print(f"Parity: {ser.parity}")
    print(f"Stopbits: {ser.stopbits}")
    print(f"XON/XOFF: {ser.xonxoff}")
    print(f"RTS/CTS: {ser.rtscts}")
    print(f"DSR/DTR: {ser.dsrdtr}")
    print(f"RTS: {ser.rts}")
    print(f"DTR: {ser.dtr}")
    print(f"CTS: {ser.cts}")
    print(f"DSR: {ser.dsr}")
    print(f"CD: {ser.cd}")
    print(f"RI: {ser.ri}")
    print()

def test_read(ser, label, timeout=2):
    """Read any available data."""
    ser.timeout = timeout
    data = ser.read(4096)
    if data:
        print(f"[{label}] Received {len(data)} bytes: {data!r}")
        try:
            print(f"[{label}] As text: {data.decode('ascii', errors='replace')}")
        except:
            pass
    else:
        print(f"[{label}] No data received (timeout={timeout}s)")
    return data

def test_write(ser, data, label):
    """Write data and check for response."""
    print(f"[{label}] Sending: {data!r}")
    written = ser.write(data)
    ser.flush()
    print(f"[{label}] Wrote {written} bytes")
    time.sleep(0.5)
    return test_read(ser, f"{label} response", timeout=2)

def main():
    print("=" * 60)
    print("ZyboGPT UART RX Diagnostic")
    print("=" * 60)

    # Check port exists
    if not os.path.exists(PORT):
        print(f"ERROR: {PORT} does not exist!")
        sys.exit(1)

    # Test 1: Open with explicit no-flow-control
    print("\n--- Test 1: Basic open, no flow control ---")
    try:
        ser = serial.Serial(
            PORT, BAUD,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False,
            timeout=2
        )
        check_port_settings(ser)

        # Read any pending data from board
        print("Reading pending data from board...")
        test_read(ser, "pending", timeout=3)

        # Try sending a newline to trigger prompt
        print("\nSending CR+LF to trigger prompt...")
        test_write(ser, b'\r\n', "newline")

        # Try sending "CONFIG\r\n" command
        print("\nSending CONFIG command...")
        test_write(ser, b'CONFIG\r\n', "config")

        ser.close()
    except serial.SerialException as e:
        print(f"Serial error: {e}")

    # Test 2: Try with DTR/RTS explicitly set
    print("\n--- Test 2: DTR=True, RTS=True ---")
    try:
        ser = serial.Serial(
            PORT, BAUD,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False,
            timeout=2
        )
        ser.dtr = True
        ser.rts = True
        time.sleep(0.1)
        print(f"CTS after RTS=True: {ser.cts}")
        print(f"DSR after DTR=True: {ser.dsr}")

        # Send CONFIG command
        test_write(ser, b'CONFIG\r\n', "config-dtr")

        ser.close()
    except serial.SerialException as e:
        print(f"Serial error: {e}")

    # Test 3: Try with RTS/CTS hardware flow control
    print("\n--- Test 3: RTS/CTS flow control enabled ---")
    try:
        ser = serial.Serial(
            PORT, BAUD,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            xonxoff=False,
            rtscts=True,  # Enable hardware flow control
            dsrdtr=False,
            timeout=2
        )
        print(f"CTS: {ser.cts}")

        test_write(ser, b'CONFIG\r\n', "config-rtscts")

        ser.close()
    except serial.SerialException as e:
        print(f"Serial error: {e}")

    # Test 4: Send individual bytes with delays
    print("\n--- Test 4: Send bytes individually with delays ---")
    try:
        ser = serial.Serial(
            PORT, BAUD,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False,
            timeout=2
        )

        msg = b'BENCH\r\n'
        print(f"Sending '{msg!r}' byte by byte...")
        for b in msg:
            ser.write(bytes([b]))
            ser.flush()
            time.sleep(0.05)  # 50ms between bytes

        time.sleep(1)
        test_read(ser, "bench-bytewise", timeout=3)

        ser.close()
    except serial.SerialException as e:
        print(f"Serial error: {e}")

    # Test 5: Use os-level write
    print("\n--- Test 5: OS-level file I/O ---")
    try:
        # Configure with stty first
        os.system(f'stty -F {PORT} {BAUD} cs8 -cstopb -parenb -crtscts -ixon -ixoff raw')
        time.sleep(0.2)

        fd = os.open(PORT, os.O_RDWR | os.O_NOCTTY)
        os.write(fd, b'CONFIG\r\n')
        time.sleep(1)
        try:
            data = os.read(fd, 4096)
            print(f"[os-write] Received: {data!r}")
        except BlockingIOError:
            print("[os-write] No data (would block)")
        os.close(fd)
    except OSError as e:
        print(f"OS error: {e}")

    print("\n" + "=" * 60)
    print("Diagnostic complete")

if __name__ == '__main__':
    main()

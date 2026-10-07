"""
teensy_check.py

Read-only health check of the Teensy + servo bus (sends no move commands).
Close the Arduino IDE Serial Monitor first (only one program can hold the port).
Port: env DELTA_PORT (default COM3 on Windows, /dev/ttyACM0 on Linux).

  python teensy_check.py
  python teensy_check.py --watch     # live status; wiggle wires one at a time to find a bad one
  python teensy_check.py --magnet    # magnet on for 3 s at the current pose (robot holds still)
  python teensy_check.py --magnet 20 # same, 20 s (time to measure with a multimeter)

Prints the firmware's debug dump (`d`), then a few streamed lines in AUTO mode with each
field labeled, then stops streaming (`x`). A follower showing -999 / nan = that servo isn't
answering: check servo power and the daisy-chain cable.
"""

import os
import sys
import time
import serial

PORT = os.environ.get("DELTA_PORT", "COM3" if sys.platform.startswith("win") else "/dev/ttyACM0")
BAUD = 115200
FIELDS = ["millis", "leader1", "leader2", "leader3", "follower1", "follower2", "follower3", "magnet"]


def read_for(ser, seconds):
    out, t_end = [], time.time() + seconds
    while time.time() < t_end:
        line = ser.readline().decode(errors="ignore").strip()
        if line:
            out.append(line)
    return out


def watch(ser):
    """Live status while you wiggle wires one at a time. Ctrl+C to stop."""
    ser.write(b"a\n")
    print("watching (Ctrl+C to stop). OK = answering, -- = dead.  Wiggle ONE wire at a time.\n")
    names = ["enc1", "enc2", "enc3", "servo1", "servo2", "servo3"]
    last = None
    while True:
        line = ser.readline().decode(errors="ignore").strip()
        vals = line.split(",")
        if len(vals) != 8:
            continue
        status = ["--" if v.strip().startswith("-999") else "OK" for v in vals[1:7]]
        if status != last:                     # print only when something changes
            stamp = time.strftime("%H:%M:%S")
            print(f"{stamp}  " + "  ".join(f"{n}={s}" for n, s in zip(names, status)), flush=True)
            last = status


def magnet_test(ser, seconds=3.0):
    """Hold the current pose and switch the magnet on for `seconds`, then off. Robot doesn't move."""
    ser.write(b"a\n")
    pose = None
    for l in read_for(ser, 1.0):
        v = l.split(",")
        if len(v) == 8 and "-999" not in "".join(v[4:7]):
            pose = v[4:7]
    if pose is None:
        print("no valid servo readings; fix the servo bus first.")
        return
    cmd = "c," + ",".join(p.strip() for p in pose)
    print(f"holding current pose {pose}. Magnet ON for {seconds:.0f} s: hold a screwdriver or the bolt under it.")
    ser.write(f"{cmd},1\n".encode())
    on = [l.split(",")[7] for l in read_for(ser, seconds) if l.count(",") == 7]
    ser.write(f"{cmd},0\n".encode())
    print(f"magnet OFF. Teensy reported magnet={sorted(set(on))} while ON was commanded "
          f"({'firmware switched it on' if '1' in on else 'firmware did NOT switch it on'}).")
    print("If the firmware switched it on but nothing pulled: the problem is the H-bridge, "
          "its power supply, or the magnet wiring (pins 30/32 + GND).")


def main():
    import sys
    ser = serial.Serial(PORT, BAUD, timeout=0.1)
    if "--magnet" in sys.argv:
        time.sleep(1.0)
        ser.reset_input_buffer()
        try:
            nums = [a for a in sys.argv[1:] if a.replace(".", "", 1).isdigit()]
            magnet_test(ser, float(nums[0]) if nums else 3.0)
        finally:
            ser.write(b"x\n")
            time.sleep(0.1)
            ser.close()
        return
    if "--watch" in sys.argv:
        time.sleep(1.0)
        ser.reset_input_buffer()
        try:
            watch(ser)
        except KeyboardInterrupt:
            pass
        finally:
            ser.write(b"x\n")
            time.sleep(0.1)
            ser.close()
        return
    time.sleep(1.0)
    ser.reset_input_buffer()
    try:
        print("--- debug dump (d) ---")
        ser.write(b"d\n")
        lines = read_for(ser, 1.0)
        print("\n".join(lines) if lines else "(nothing: firmware not running or hung -> replug USB / re-upload)")

        print("\n--- streamed lines in AUTO (a), no move commands sent ---")
        ser.write(b"a\n")
        lines = [l for l in read_for(ser, 1.0) if l.count(",") == 7][:5]
        if not lines:
            print("(no stream lines)")
        for l in lines:
            vals = l.split(",")
            print("  " + "  ".join(f"{k}={v}" for k, v in zip(FIELDS, vals)))
        if lines:
            dead = [k for k, v in zip(FIELDS[4:7], lines[-1].split(",")[4:7]) if v.strip() in ("-999", "-999.00", "nan")]
            print("\nservos not answering: " + (", ".join(dead) if dead else "none, all three OK"))

        stream = [l for l in read_for(ser, 2.0) if l.count(",") == 7]
        ms = [int(l.split(",")[0]) for l in stream if l.split(",")[0].isdigit()]
        if len(ms) > 2:
            hz = 1000.0 * (len(ms) - 1) / (ms[-1] - ms[0])
            gap = max(b - a for a, b in zip(ms, ms[1:]))
            verdict = "OK" if hz > 45 else "TOO SLOW (firmware loop is stalling)"
            print(f"stream rate: {hz:.0f} lines/s (should be ~60), longest gap {gap} ms  -> {verdict}")
    finally:
        ser.write(b"x\n")
        time.sleep(0.1)
        ser.close()


if __name__ == "__main__":
    main()

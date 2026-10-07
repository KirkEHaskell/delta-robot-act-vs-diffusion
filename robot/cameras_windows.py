"""
Resolve cameras by stable USB port ID -> current OpenCV index or device path.

Windows hands OpenCV/DirectShow indices in enumeration order, which shuffles
whenever a camera drops/replugs -- so an index like "1" is NOT a stable identity.
Each camera's DevicePath, however, encodes its physical USB port, which IS stable
(same across replug and reboot; only changes if you physically move the camera to
another port -- specifically to a different slot on the 1->4 expander hub; the
laptop C port it plugs into does NOT matter). We map port IDs -> role keys once,
then look up the current index/path for each port at startup.

First-time setup on your machine: run `python cameras_windows.py`, note the port ID printed for
each camera, and edit CAMERA_MAP below so each port maps to its role (side / top_right / top_left).

Colour note: in this study the side and top_right cameras ran with auto white balance OFF and a
fixed exposure, left over from an old OpenCV script that set CAP_PROP_AUTO_WB=0 and
CAP_PROP_EXPOSURE=-6 (Windows remembers those settings per device). All training data has that
yellow-green cast, so the published checkpoints expect it. See the README, "Camera colour".
"""

from pygrabber.dshow_graph import (IPropertyBag, GUID, DeviceCategories,
                                   clsids, ICreateDevEnum, client)

# Stable USB port ID  ->  camera/dataset key.   *** EDIT FOR YOUR CAMERAS ***
CAMERA_MAP = {
    "7&2bd3af6b&0&0000": "side",       # port A
    "8&e608a48&0&0000":  "top_right",  # port B
    "8&2487f305&0&0000": "top_left",   # port C
}


def _read_prop(moniker, name):
    try:
        pb = moniker.BindToStorage(0, 0, IPropertyBag._iid_).QueryInterface(IPropertyBag)
        return pb.Read(name, pErrorLog=None) or ""
    except Exception:
        return ""


def enumerate_video_devices():
    """[(index, friendly_name, device_path)] in OpenCV DSHOW index order."""
    sde = client.CreateObject(clsids.CLSID_SystemDeviceEnum, interface=ICreateDevEnum)
    enum = sde.CreateClassEnumerator(GUID(DeviceCategories.VideoInputDevice), dwFlags=0)
    out, idx = [], 0
    try:
        moniker, count = enum.Next(1)
    except ValueError:
        return out
    while count > 0:
        out.append((idx, _read_prop(moniker, "FriendlyName").strip(),
                    _read_prop(moniker, "DevicePath")))
        idx += 1
        moniker, count = enum.Next(1)
    return out


def port_id(device_path):
    r"""\\?\usb#vid_0c45&pid_6367&mi_00#<PORTID>#{guid}\global -> <PORTID>"""
    parts = device_path.split("#")
    return parts[2] if len(parts) > 2 else device_path


def av_input(device_path):
    """PyAV/ffmpeg dshow input string for a device path (addresses the camera by
    its stable USB port, and lets us force the MJPG pin -- which OpenCV cannot)."""
    return "video=@device_pnp_" + device_path


def _resolve(camera_map):
    """{port_id: key} -> {key: (index, device_path)}. Fail loud (RuntimeError) if any
    mapped camera is missing or if two resolve to the same device -- so you can never
    silently record a scrambled or partial camera set."""
    devices = enumerate_video_devices()
    by_port = {port_id(path): (idx, path) for idx, _, path in devices}

    resolved, missing = {}, []
    for pid, key in camera_map.items():
        if pid in by_port:
            resolved[key] = by_port[pid]
        else:
            missing.append((key, pid))

    if missing:
        want = "\n".join(f"    {key}: port {pid}  -- NOT FOUND" for key, pid in missing)
        have = "\n".join(f"    index {i}: {n}  [{port_id(p)}]" for i, n, p in devices) or "    (none)"
        raise RuntimeError(
            "Camera resolve failed. These mapped cameras are not plugged into their "
            f"ports:\n{want}\n  Cameras available right now:\n{have}\n"
            "  Fix: plug each camera into its assigned expander slot (or update CAMERA_MAP).")

    seen = {}
    for key, (idx, _) in resolved.items():
        if idx in seen:
            raise RuntimeError(f"Cameras {seen[idx]!r} and {key!r} both resolved to index "
                               f"{idx} -- two CAMERA_MAP entries point at the same device.")
        seen[idx] = key
    return resolved


def resolve_cameras(camera_map=CAMERA_MAP):
    """{port_id: key} -> {key: opencv_index}."""
    return {key: idx for key, (idx, _) in _resolve(camera_map).items()}


def resolve_camera_paths(camera_map=CAMERA_MAP):
    """{port_id: key} -> {key: device_path} (for PyAV via av_input())."""
    return {key: path for key, (_, path) in _resolve(camera_map).items()}


def apply_controls(key, dev, match_training=True):
    """No-op on Windows: DirectShow keeps each camera's last settings (see the module docstring)."""


def av_open_args(dev):
    """(file, kwargs) for av.open(): MJPG at 320x240@30 through DirectShow, addressed by USB port."""
    return av_input(dev), {"format": "dshow",
                           "options": {"vcodec": "mjpeg", "video_size": "320x240", "framerate": "30"}}


if __name__ == "__main__":
    print("Video devices seen now:")
    for i, n, p in enumerate_video_devices():
        print(f"  index {i}: {n}  [{port_id(p)}]")
    print("\nResolved role -> index / path:")
    for key, (idx, path) in _resolve(CAMERA_MAP).items():
        print(f"  {key}: index {idx}")
        print(f"        {av_input(path)}")

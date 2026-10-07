"""
Pick the camera backend for this OS. Both backends expose the same interface:

  CAMERA_MAP                       {usb_port_id: role}, roles = side / top_right / top_left
  resolve_camera_paths()           {role: device}; fails loud if a camera is missing
  apply_controls(role, dev, match_training=True)
  av_open_args(dev)                (file, kwargs) for av.open(): MJPG 320x240 @ 30 fps

Cameras are addressed by the physical USB port they are plugged into, never by OS index (indices
reshuffle whenever a camera replugs). Edit CAMERA_MAP in cameras_windows.py / cameras_linux.py.
"""

import sys

if sys.platform.startswith("win"):
    from cameras_windows import CAMERA_MAP, apply_controls, av_open_args, resolve_camera_paths  # noqa: F401
else:
    from cameras_linux import CAMERA_MAP, apply_controls, av_open_args, resolve_camera_paths  # noqa: F401

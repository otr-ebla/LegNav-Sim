"""
make_tb4_urdf.py — Flatten the official TurtleBot4 (standard) xacro into a
plain URDF that habitat-sim can load, without a ROS installation.

The upstream xacros resolve files through ROS (`$(find pkg)` and
`package://pkg/...`). Here both are rewritten to absolute paths on disk, then
xacro is expanded and all Gazebo-only elements are stripped (habitat-sim's
URDF importer ignores them anyway, but they bloat the file).

Expected layout (clone once):
  <root>/turtlebot4    git clone -b humble https://github.com/turtlebot/turtlebot4
  <root>/create3_sim   git clone -b humble https://github.com/iRobotEducation/create3_sim

Run (in the `habitat` conda env, which has the `xacro` pip package):
  python -m legnav.habitat_eval.make_tb4_urdf --root ~/habitat_data/robots
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET


def package_dirs(root: str) -> dict:
    c3 = os.path.join(root, "create3_sim", "irobot_create_common")
    return {
        "turtlebot4_description": os.path.join(root, "turtlebot4", "turtlebot4_description"),
        "irobot_create_description": os.path.join(c3, "irobot_create_description"),
        "irobot_create_control": os.path.join(c3, "irobot_create_control"),
    }


def rewrite_tree(src_pkgs: dict, tmp: str) -> dict:
    """Copy each package's urdf/ tree to tmp with ROS paths made absolute."""
    pat_find = re.compile(r"\$\(find ([A-Za-z0-9_]+)\)")
    pat_pkg = re.compile(r"package://([A-Za-z0-9_]+)")
    out = {}
    for name, path in src_pkgs.items():
        dst = os.path.join(tmp, name)
        out[name] = dst
        if not os.path.isdir(path):
            continue
        for dirpath, _, files in os.walk(path):
            if "/.git" in dirpath:
                continue
            rel = os.path.relpath(dirpath, path)
            os.makedirs(os.path.join(dst, rel), exist_ok=True)
            for f in files:
                s, d = os.path.join(dirpath, f), os.path.join(dst, rel, f)
                if f.endswith((".xacro", ".urdf")):
                    txt = open(s).read()
                    # $(find pkg) → rewritten copy (xacro includes);
                    # package://pkg → ORIGINAL package (meshes stay in place).
                    txt = pat_find.sub(lambda m: os.path.join(tmp, m.group(1)), txt)
                    txt = pat_pkg.sub(lambda m: src_pkgs.get(m.group(1), m.group(0)), txt)
                    open(d, "w").write(txt)
                else:
                    os.symlink(s, d)
    return out


def strip_gazebo(urdf_xml: str) -> str:
    root = ET.fromstring(urdf_xml)
    for tag in ("gazebo", "transmission", "ros2_control", "plugin"):
        for parent in root.iter():
            for child in list(parent):
                if child.tag == tag:
                    parent.remove(child)
    # Mesh filenames: drop any leftover file:// scheme.
    for mesh in root.iter("mesh"):
        fn = mesh.get("filename", "")
        if fn.startswith("file://"):
            mesh.set("filename", fn[len("file://"):])
    ET.indent(root)
    return ET.tostring(root, encoding="unicode")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/habitat_data/robots"))
    ap.add_argument("--out", default=None, help="default: <root>/turtlebot4.urdf")
    args = ap.parse_args()
    out = args.out or os.path.join(args.root, "turtlebot4.urdf")

    pkgs = package_dirs(args.root)
    with tempfile.TemporaryDirectory() as tmp:
        tpk = rewrite_tree(pkgs, tmp)
        top = os.path.join(tpk["turtlebot4_description"], "urdf", "standard", "turtlebot4.urdf.xacro")
        xacro_bin = shutil.which("xacro") or os.path.join(os.path.dirname(sys.executable), "xacro")
        res = subprocess.run([xacro_bin, top, "gazebo:=classic", "namespace:=", "visualize_rays:=false"],
                             capture_output=True, text=True)
        if res.returncode != 0:
            sys.exit(f"xacro failed:\n{res.stderr}")
        urdf = strip_gazebo(res.stdout)

    open(out, "w").write(urdf)
    meshes = re.findall(r'filename="([^"]+)"', urdf)
    missing = [m for m in meshes if not os.path.exists(m)]
    print(f"wrote {out}  links={urdf.count('<link ')}  meshes={len(meshes)}  missing={len(missing)}")
    for m in missing:
        print("  missing:", m)


if __name__ == "__main__":
    main()

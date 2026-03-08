def parse_asf(asf_path):
    """
    Parse a .asf skeleton file.

    Returns:
        bones     : dict  { joint_name -> { direction, length, dof, axis } }
        hierarchy : dict  { joint_name -> parent_name }   (root has no entry)
        order     : list  joint names in definition order
    """
    bones = {}
    hierarchy = {}   # child -> parent
    order = []

    with open(asf_path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # ── bonedata block ──────────────────────────────────────────────────
        if line == ":bonedata":
            i += 1
            while i < len(lines) and lines[i].strip() != ":hierarchy":
                if lines[i].strip() == "begin":
                    bone = {
                        "direction": [0.0, 1.0, 0.0],
                        "length":    1.0,
                        "dof":       [],
                        "axis":      [0.0, 0.0, 0.0],
                    }
                    name = None
                    i += 1
                    while i < len(lines) and lines[i].strip() != "end":
                        parts = lines[i].strip().split()
                        if not parts:
                            i += 1
                            continue
                        key = parts[0]
                        if key == "name":
                            name = parts[1]
                        elif key == "direction":
                            bone["direction"] = list(map(float, parts[1:4]))
                        elif key == "length":
                            bone["length"] = float(parts[1])
                        elif key == "dof":
                            bone["dof"] = parts[1:]
                        elif key == "axis":
                            bone["axis"] = list(map(float, parts[1:4]))
                        i += 1
                    if name:
                        bones[name] = bone
                        order.append(name)
                i += 1
            continue   # skip the i += 1 at bottom, we're already past bonedata

        # ── hierarchy block ─────────────────────────────────────────────────
        if line == ":hierarchy":
            i += 1
            while i < len(lines) and lines[i].strip() != "end":
                parts = lines[i].strip().split()
                if len(parts) >= 2:
                    parent = parts[0]
                    for child in parts[1:]:
                        hierarchy[child] = parent
                i += 1

        i += 1

    return bones, hierarchy
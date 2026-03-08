def parse_asf(asf_path):
    joints = {}
    hierarchy = []

    with open(asf_path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line == ":bonedata":
            i += 1
            while lines[i].strip() != ":hierarchy":
                if lines[i].strip() == "begin":
                    joint = {}
                    i += 1
                    while lines[i].strip() != "end":
                        parts = lines[i].strip().split()
                        if parts[0] == "name":
                            joint_name = parts[1]
                        i += 1
                    joints[joint_name] = joint
                i += 1

        # Parse hierarchy
        if line == ":hierarchy":
            i += 1
            while lines[i].strip() != "end":
                parts = lines[i].strip().split()
                parent = parts[0]
                children = parts[1:]
                hierarchy.append((parent, children))
                i += 1

        i += 1

    return joints, hierarchy
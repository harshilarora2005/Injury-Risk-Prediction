def parse_asf_geometry(asf_path):
    bones = {}

    with open(asf_path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line == ":bonedata":
            i += 1
            while lines[i].strip() != ":hierarchy":
                if lines[i].strip() == "begin":
                    name = None
                    direction = None
                    length = None

                    i += 1
                    while lines[i].strip() != "end":
                        parts = lines[i].strip().split()

                        if parts[0] == "name":
                            name = parts[1]

                        elif parts[0] == "direction":
                            direction = list(map(float, parts[1:]))

                        elif parts[0] == "length":
                            length = float(parts[1])

                        i += 1

                    bones[name] = {
                        "direction": direction,
                        "length": length
                    }

                i += 1

        i += 1

    return bones
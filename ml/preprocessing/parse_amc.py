def is_float(token):
    try:
        float(token)
        return True
    except ValueError:
        return False


def parse_amc(amc_path):
    """
    Returns:
        frames: list of dicts
            frames[t][joint_name] = list of angle values
    """
    frames = []
    current_frame = None

    with open(amc_path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()

            # Skip empty lines and metadata
            if not line or line.startswith(":"):
                continue

            # New frame number
            if line.isdigit():
                if current_frame is not None:
                    frames.append(current_frame)
                current_frame = {}
                continue

            parts = line.split()

            if len(parts) < 2:
                continue

            joint_name = parts[0]
            value_tokens = parts[1:]

            if not all(is_float(v) for v in value_tokens):
                continue

            values = list(map(float, value_tokens))
            current_frame[joint_name] = values

    if current_frame is not None:
        frames.append(current_frame)

    return frames
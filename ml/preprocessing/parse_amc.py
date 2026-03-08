def parse_amc(amc_path):
    """
    Parse a .amc motion capture file.

    Returns
    -------
    motions : list of dicts  { joint_name -> [float, ...] }
        One dict per frame, values are raw angle data from the file.
    """
    motions = []
    current = None

    with open(amc_path, 'r', errors='ignore') as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith(':') or line.startswith('#'):
                continue

            # Frame number — start a new frame dict
            if line.isdigit():
                if current is not None:
                    motions.append(current)
                current = {}
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            try:
                values = list(map(float, parts[1:]))
                current[parts[0]] = values
            except ValueError:
                continue

    if current is not None:
        motions.append(current)

    return motions
def convert_hevc_cra_to_bla(packet_data: bytes) -> bytes:
    """
    Convert CRA (Clean Random Access) frames to BLA (Broken Link Access) frames in H.265/HEVC packet data.
    CRA frames (NAL type 21) are converted to BLA_W_LP frames (NAL type 16).

    This is necessary when cutting video because CRA frames may have RASL pictures that reference
    frames before the CRA, making them unsuitable as random access points in cut videos.
    BLA frames guarantee no pictures after them reference pictures before them.

    Returns the modified packet data, or original data if no conversion was needed.
    """
    if not packet_data or len(packet_data) < 6:
        return packet_data

    data = bytearray(packet_data)
    modified = False

    # Handle MP4/ISOBMFF format first (4-byte length prefix)
    if len(data) >= 6:
        nal_length = int.from_bytes(data[:4], byteorder='big')
        if nal_length > 4 and nal_length <= len(data) - 4:
            # Found valid length-prefixed NAL unit
            nal_header_pos = 4
            if nal_header_pos + 1 < len(data):
                nal_type = (data[nal_header_pos] >> 1) & 0x3F
                if nal_type == 21:  # CRA frame
                    # Convert to BLA_W_LP (type 16)
                    # Clear bits 1-6 and set to 16 (0010000)
                    data[nal_header_pos] = (data[nal_header_pos] & 0x81) | (16 << 1)
                    modified = True

    # Handle Annex B format (start codes) - convert all CRA frames found
    if not modified:  # Only process if not already handled as MP4
        i = 0
        while i < len(data) - 5:
            if data[i:i+4] == b'\x00\x00\x00\x01':
                if i + 6 <= len(data):
                    nal_header_pos = i + 4
                    nal_type = (data[nal_header_pos] >> 1) & 0x3F
                    if nal_type == 21:  # CRA frame
                        # Convert to BLA_W_LP (type 16)
                        data[nal_header_pos] = (data[nal_header_pos] & 0x81) | (16 << 1)
                        modified = True
                i += 1
            elif data[i:i+3] == b'\x00\x00\x01':
                if i + 5 <= len(data):
                    nal_header_pos = i + 3
                    nal_type = (data[nal_header_pos] >> 1) & 0x3F
                    if nal_type == 21:  # CRA frame
                        # Convert to BLA_W_LP (type 16)
                        data[nal_header_pos] = (data[nal_header_pos] & 0x81) | (16 << 1)
                        modified = True
                i += 1
            else:
                i += 1

    return bytes(data)


def get_h265_nal_unit_type(packet_data: bytes) -> int | None:
    """
    Extract NAL unit type from H.265/HEVC packet data.
    For packets with multiple NAL units, prioritizes picture NAL types (0-21)
    over metadata types (32-40). Returns safe keyframes first (16-20), then
    other picture types, then metadata.

    H.265 NAL unit types:
    - 0-21: Picture NAL types (actual video data, priority over metadata)
    - 16-18: BLA frames (safe cut points)
    - 19, 20: IDR frames (safe cut points)
    - 21: CRA frame (not safe for cutting due to RASL pictures)
    - 32-34: VPS, SPS, PPS (parameter sets)
    - 35: AUD (Access Unit Delimiter)
    """
    if not packet_data or len(packet_data) < 6:
        return None

    data = bytes(packet_data)

    # H.265 in MP4 containers uses length-prefixed NAL units, not Annex B start codes
    # Try MP4/ISOBMFF format first (4-byte length prefix)
    # But avoid false positive detection of Annex B start codes (0x00000001)
    if len(data) >= 6:
        # Read the first NAL unit length (big-endian 4 bytes)
        nal_length = int.from_bytes(data[:4], byteorder='big')
        # Avoid misinterpreting Annex B start codes as MP4 lengths
        # Annex B start codes are 0x00000001 or 0x000001, which would be lengths 1 or very small
        if nal_length > 4 and nal_length <= len(data) - 4:
            # Found valid length-prefixed NAL unit
            nal_header = data[4:6]
            nal_unit_type = (nal_header[0] >> 1) & 0x3F
            return nal_unit_type

    # Try Annex B format (start codes) - search entire packet for safe keyframes
    nal_types_found = []
    i = 0
    while i < len(data) - 5:  # H.265 needs 2 bytes for NAL header
        if data[i:i+4] == b'\x00\x00\x00\x01':
            if i + 6 <= len(data):
                nal_header = data[i+4:i+6]
                nal_type = (nal_header[0] >> 1) & 0x3F
                nal_types_found.append(nal_type)
                # Found safe keyframe - prioritize these
                if nal_type in [16, 17, 18, 19, 20]:  # BLA or IDR frames
                    return nal_type
            i += 4
        elif data[i:i+3] == b'\x00\x00\x01':
            if i + 5 <= len(data):
                nal_header = data[i+3:i+5]
                nal_type = (nal_header[0] >> 1) & 0x3F
                nal_types_found.append(nal_type)
                # Found safe keyframe - prioritize these
                if nal_type in [16, 17, 18, 19, 20]:  # BLA or IDR frames
                    return nal_type
            i += 3
        else:
            i += 1

    # No safe keyframes found, prioritize picture types (0-21) over metadata (32-40)
    if nal_types_found:
        # First check for CRA frames (21) - these are picture types but need special handling
        for nal_type in nal_types_found:
            if nal_type == 21:  # CRA frame
                return nal_type

        # Then check for any other picture NAL types (0-15)
        for nal_type in nal_types_found:
            if 0 <= nal_type <= 15:  # Other picture types
                return nal_type

        # Finally return first metadata type if no pictures found
        return nal_types_found[0]

    return None


def is_safe_h264_keyframe_nal(nal_type: int | None) -> bool:
    """
    Check if an H.264 NAL type represents a safe keyframe for cutting.

    Args:
        nal_type: H.264 NAL unit type (int)

    Returns:
        bool: True if this NAL type is safe for cutting
    """
    if nal_type is None:
        return True # Can't know for sure
    # Accept IDR frames (5), SEI (6), and parameter sets (7,8) as cutting points
    return nal_type in [5, 6, 7, 8]


def is_safe_h265_keyframe_nal(nal_type: int | None) -> bool:
    """
    Check if an H.265 NAL type represents a safe keyframe for cutting.

    Args:
        nal_type: H.265 NAL unit type (int)

    Returns:
        bool: True if this NAL type is safe for cutting
    """
    if nal_type is None:
        return True  # Can't know for sure
    # Accept BLA(16,17,18), IDR(19,20), CRA(21) frames and parameter sets (32,33,34)
    return nal_type in [16, 17, 18, 19, 20, 21, 32, 33, 34]

def get_h264_nal_unit_type(packet_data: bytes) -> int | None:
    """
    Extract NAL unit type from H.264/AVC packet data.
    For packets with multiple NAL units, prioritizes picture NAL types (1-5)
    over metadata types (6-9). Returns type 5 (IDR) if found, otherwise
    returns the most important picture type, or first metadata type if no pictures.

    H.264 NAL unit types:
    - 5: IDR frame (safe cut point)
    - 1-4: Non-IDR slices (picture data, priority over metadata)
    - 7, 8: SPS, PPS (parameter sets)
    - 9: AUD (Access Unit Delimiter)
    """
    if not packet_data or len(packet_data) < 5:
        return None

    data = bytes(packet_data)

    # H.264 in MP4 containers uses length-prefixed NAL units, not Annex B start codes
    # Try MP4/ISOBMFF format first (4-byte length prefix)
    # But avoid false positive detection of Annex B start codes (0x00000001)
    if len(data) >= 5:
        # Read the first NAL unit length (big-endian 4 bytes)
        nal_length = int.from_bytes(data[:4], byteorder='big')
        # Avoid misinterpreting Annex B start codes as MP4 lengths
        # Annex B start codes are 0x00000001 or 0x000001, which would be lengths 1 or very small
        if nal_length > 4 and nal_length <= len(data) - 4:
            # Found valid length-prefixed NAL unit
            nal_header = data[4]
            nal_unit_type = nal_header & 0x1F  # H.264 uses lower 5 bits
            return nal_unit_type

    # Try Annex B format (start codes) - search for best NAL type
    nal_types_found = []
    i = 0
    while i < len(data) - 4:
        if data[i:i+4] == b'\x00\x00\x00\x01':
            if i + 4 < len(data):
                nal_header = data[i+4]
                nal_type = nal_header & 0x1F
                nal_types_found.append(nal_type)
                if nal_type == 5:  # Found IDR frame - highest priority!
                    return 5
            i += 4
        elif data[i:i+3] == b'\x00\x00\x01':
            if i + 3 < len(data):
                nal_header = data[i+3]
                nal_type = nal_header & 0x1F
                nal_types_found.append(nal_type)
                if nal_type == 5:  # Found IDR frame - highest priority!
                    return 5
            i += 3
        else:
            i += 1

    # No IDR found, prioritize picture types (1-4) over metadata types (6-9)
    if nal_types_found:
        # Look for any picture NAL types first
        for nal_type in nal_types_found:
            if 1 <= nal_type <= 4:  # Non-IDR picture types
                return nal_type
        # If no picture types, return first metadata type
        return nal_types_found[0]

    return None

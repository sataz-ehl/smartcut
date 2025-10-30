#!/usr/bin/env python3
"""
Detect H.264 SPS ids used in an MP4 file by:
- Parsing avcC (extradata) to extract SPS NAL units and their seq_parameter_set_id
- Scanning the first few video packets for in-band SPS and extracting their ids

Usage:
  python scripts/detect_h264_sps_ids.py [path/to/video.mp4]

Prints the sets of SPS ids found in extradata, in-band samples, and their union.
"""
import sys
import struct
from typing import List, Set

import av


def ebsp_to_rbsp(data: bytes) -> bytes:
    out = bytearray()
    zeros = 0
    i = 0
    while i < len(data):
        b = data[i]
        if zeros >= 2 and b == 0x03:
            # skip emulation prevention byte
            i += 1
            zeros = 0
            continue
        out.append(b)
        if b == 0x00:
            zeros += 1
        else:
            zeros = 0
        i += 1
    return bytes(out)


class BitReader:
    def __init__(self, data: bytes):
        self.data = data
        self.bitpos = 0

    def bits_left(self) -> int:
        return 8 * len(self.data) - self.bitpos

    def read_bits(self, n: int) -> int:
        val = 0
        for _ in range(n):
            if self.bitpos >= 8 * len(self.data):
                raise EOFError("Not enough bits")
            byte = self.data[self.bitpos // 8]
            shift = 7 - (self.bitpos % 8)
            bit = (byte >> shift) & 1
            val = (val << 1) | bit
            self.bitpos += 1
        return val

    def read_bit(self) -> int:
        return self.read_bits(1)

    def read_ue(self) -> int:
        zeros = 0
        while self.bits_left() > 0 and self.read_bit() == 0:
            zeros += 1
        if zeros == 0:
            return 0
        code_num = (1 << zeros) - 1 + self.read_bits(zeros)
        return code_num


def parse_h264_sps_id(nal_payload: bytes) -> int:
    """
    Parse seq_parameter_set_id from an H.264 SPS NAL payload.
    Expects nal_payload to include the 1-byte NAL header at the start.
    """
    if not nal_payload or len(nal_payload) < 4:
        raise ValueError("SPS too short")
    # Skip 1-byte NAL header
    rbsp = ebsp_to_rbsp(nal_payload[1:])
    br = BitReader(rbsp)
    # profile_idc (8), constraint flags + reserved (8), level_idc (8)
    _profile_idc = br.read_bits(8)
    _constraints = br.read_bits(8)
    _level_idc = br.read_bits(8)
    sps_id = br.read_ue()
    return sps_id


def parse_avcc_sps_ids(extradata: bytes) -> List[bytes]:
    """
    Return list of raw SPS NAL units (including 1-byte NAL header) parsed from avcC extradata.
    """
    if not extradata or len(extradata) < 6:
        return []
    data = extradata
    # AVCDecoderConfigurationRecord structure
    # configurationVersion (1)
    # AVCProfileIndication (1)
    # profile_compatibility (1)
    # AVCLevelIndication (1)
    # lengthSizeMinusOne (1)
    # numOfSequenceParameterSets (1 low 5 bits)
    off = 0
    _configurationVersion = data[off]; off += 1
    _profile = data[off]; off += 1
    _compat = data[off]; off += 1
    _level = data[off]; off += 1
    _length_size_minus_one = data[off] & 0x3; off += 1
    num_sps = data[off] & 0x1F; off += 1
    sps_list: List[bytes] = []
    for _ in range(num_sps):
        if off + 2 > len(data):
            break
        (sps_len,) = struct.unpack_from(">H", data, off)
        off += 2
        if off + sps_len > len(data):
            break
        sps = data[off:off + sps_len]
        off += sps_len
        sps_list.append(sps)
    # Skip PPS records if present (not needed for SPS id discovery)
    return sps_list


def detect_sps_ids(path: str, scan_packets: bool = True, max_packets: int = 200) -> tuple[Set[int], Set[int]]:
    extradata_ids: Set[int] = set()
    inband_ids: Set[int] = set()

    with av.open(path, 'r', metadata_errors='ignore') as c:
        if not c.streams.video:
            return extradata_ids, inband_ids
        v = c.streams.video[0]
        # From avcC extradata
        try:
            if v.codec_context.extradata:
                sps_list = parse_avcc_sps_ids(v.codec_context.extradata)
                for sps in sps_list:
                    try:
                        sps_id = parse_h264_sps_id(sps)
                        extradata_ids.add(sps_id)
                    except Exception:
                        pass
        except Exception:
            pass

        if not scan_packets:
            return extradata_ids, inband_ids

        # Scan first packets for in-band SPS
        count = 0
        for pkt in c.demux(v):
            if pkt.pts is None:
                continue
            data = bytes(pkt)
            off = 0
            # AVCC format: 4-byte big-endian NAL length
            while off + 4 <= len(data):
                nal_len = int.from_bytes(data[off:off + 4], 'big')
                off += 4
                if nal_len <= 0 or off + nal_len > len(data):
                    break
                nal = data[off:off + nal_len]
                off += nal_len
                if not nal:
                    continue
                nal_type = nal[0] & 0x1F
                if nal_type == 7:  # SPS
                    try:
                        sps_id = parse_h264_sps_id(nal)
                        inband_ids.add(sps_id)
                    except Exception:
                        pass
            count += 1
            if count >= max_packets:
                break

    return extradata_ids, inband_ids


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else 'tests/test_data/google_subaru.mp4'
    extradata_ids, inband_ids = detect_sps_ids(path)
    all_ids = sorted(set().union(extradata_ids, inband_ids))

    print(f"Input: {path}")
    print(f"SPS ids (avcC extradata): {sorted(extradata_ids)}")
    print(f"SPS ids (in-band first packets): {sorted(inband_ids)}")
    print(f"SPS ids (union): {all_ids}")


if __name__ == '__main__':
    main()


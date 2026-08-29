#!/usr/bin/env python3
"""
Phase 4 - AIS decoding and gap analysis ("dark ship" tracking).

WHAT AIS IS

Ships broadcast position, speed and identity over VHF in a bit-packed format
(ITU-R M.1371), wrapped in NMEA !AIVDM sentences. This file decodes those
sentences - the 6-bit ASCII armouring, multi-part reassembly, position reports
(types 1/2/3/18) and static/voyage data (type 5) - and then looks for vessels
that stop transmitting.

READ THIS BEFORE USING THE GAP DETECTOR

A gap in AIS coverage is NOT evidence of wrongdoing. The innocent explanations
are numerous and ordinary:

  * satellite revisit gaps - polar-orbiting receivers simply are not overhead
  * no terrestrial receiver within VHF range, which is most of the ocean
  * message collision in congested waters, where reception degrades badly
  * transponder failure, power loss, or a genuine equipment fault
  * legitimate safety-based silence, which the regulations permit

Deliberately switching off to conceal a transfer is one item on a long list, and
it is not the most common one. So this module returns LEADS FOR INVESTIGATION.
Its vocabulary is deliberately flat - `gap`, `unexplained`, `co_located` - and
never `illegal`, `smuggling`, or `sanctioned`. Publishing an accusation about a
named vessel on the strength of a coverage hole would be both analytically
illiterate and defamatory, and no downstream consumer of this data should treat
a flag here as a finding.

A real dark-fleet analysis corroborates: SAR or optical imagery at the gap
location, port call records, draught changes on either side of the gap,
ownership and flag history. This file provides the first filter, nothing more.

stdlib + numpy. No feed is included - AIS data comes from a provider you have a
relationship with, and their terms are yours to honour.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

EARTH_RADIUS_NM = 3440.065
POSITION_TYPES = frozenset({1, 2, 3, 18})


class AisDecodeError(ValueError):
    """A sentence could not be decoded. Never guessed at."""


# ── 6-bit payload handling ────────────────────────────────────────────────────

def sixbit_to_bits(payload: str, fill_bits: int = 0) -> str:
    """
    Unpack AIS payload armouring into a bit string.

    Each character carries six bits: subtract 48, and subtract another 8 if the
    result exceeds 40. `fill_bits` says how many bits of the last character are
    padding and must be dropped.
    """
    bits: list[str] = []
    for character in payload:
        value = ord(character) - 48
        if value > 40:
            value -= 8
        if not 0 <= value < 64:
            raise AisDecodeError(f"character {character!r} is not valid AIS armouring")
        bits.append(format(value, "06b"))
    joined = "".join(bits)
    return joined[:len(joined) - fill_bits] if fill_bits else joined


def bits_to_sixbit(bits: str) -> tuple[str, int]:
    """Inverse armouring. Exists so the decoder can be tested round-trip."""
    fill = (-len(bits)) % 6
    padded = bits + "0" * fill
    out = []
    for i in range(0, len(padded), 6):
        value = int(padded[i:i + 6], 2)
        out.append(chr(value + 48 if value < 40 else value + 56))
    return "".join(out), fill


def unsigned(bits: str, start: int, length: int) -> int:
    chunk = bits[start:start + length]
    if len(chunk) < length:
        raise AisDecodeError(f"needed {length} bits at {start}, found {len(chunk)}")
    return int(chunk, 2)


def signed(bits: str, start: int, length: int) -> int:
    value = unsigned(bits, start, length)
    return value - (1 << length) if value >= (1 << (length - 1)) else value


def nmea_checksum(sentence: str) -> str:
    """XOR of everything between '!' and '*'. Two uppercase hex digits."""
    body = sentence.split("*")[0].lstrip("!$")
    checksum = 0
    for character in body:
        checksum ^= ord(character)
    return format(checksum, "02X")


# ── sentences ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AivdmSentence:
    fragment_count: int
    fragment_number: int
    sequence_id: str
    channel: str
    payload: str
    fill_bits: int
    checksum_ok: bool


def parse_aivdm(line: str) -> AivdmSentence:
    """Parse one !AIVDM sentence. Malformed input raises rather than returning junk."""
    line = line.strip()
    if not line.startswith(("!AIVDM", "!AIVDO", "!BSVDM")):
        raise AisDecodeError(f"not an AIVDM sentence: {line[:24]!r}")
    body, _, given_checksum = line.partition("*")
    parts = body.split(",")
    if len(parts) < 7:
        raise AisDecodeError(f"expected 7 comma-separated fields, found {len(parts)}")
    try:
        return AivdmSentence(
            fragment_count=int(parts[1]),
            fragment_number=int(parts[2]),
            sequence_id=parts[3],
            channel=parts[4],
            payload=parts[5],
            fill_bits=int(parts[6]) if parts[6] else 0,
            checksum_ok=(not given_checksum
                         or given_checksum.strip().upper() == nmea_checksum(line)),
        )
    except ValueError as e:
        raise AisDecodeError(f"malformed AIVDM fields: {e}") from e


@dataclass(frozen=True)
class Position:
    """One decoded position report."""

    mmsi: int
    timestamp: float
    latitude: float
    longitude: float
    sog_knots: float | None
    cog_deg: float | None
    heading_deg: float | None
    nav_status: int
    message_type: int

    @property
    def position_valid(self) -> bool:
        """AIS uses 91/181 for 'not available'. Those are not coordinates."""
        return abs(self.latitude) <= 90.0 and abs(self.longitude) <= 180.0

    @property
    def under_way(self) -> bool:
        return self.sog_knots is not None and self.sog_knots > 0.5


def decode_position(bits: str, timestamp: float = 0.0) -> Position:
    """Decode a type 1/2/3 or 18 position report from an unpacked bit string."""
    message_type = unsigned(bits, 0, 6)
    if message_type not in POSITION_TYPES:
        raise AisDecodeError(f"message type {message_type} is not a position report")
    mmsi = unsigned(bits, 8, 30)

    if message_type == 18:                      # class B has a different layout
        sog_raw = unsigned(bits, 46, 10)
        longitude_raw = signed(bits, 57, 28)
        latitude_raw = signed(bits, 85, 27)
        cog_raw = unsigned(bits, 112, 12)
        heading_raw = unsigned(bits, 124, 9)
        nav_status = 15
    else:
        nav_status = unsigned(bits, 38, 4)
        sog_raw = unsigned(bits, 50, 10)
        longitude_raw = signed(bits, 61, 28)
        latitude_raw = signed(bits, 89, 27)
        cog_raw = unsigned(bits, 116, 12)
        heading_raw = unsigned(bits, 128, 9)

    return Position(
        mmsi=mmsi,
        timestamp=timestamp,
        latitude=latitude_raw / 600_000.0,
        longitude=longitude_raw / 600_000.0,
        sog_knots=None if sog_raw == 1023 else sog_raw / 10.0,
        cog_deg=None if cog_raw == 3600 else cog_raw / 10.0,
        heading_deg=None if heading_raw == 511 else float(heading_raw),
        nav_status=nav_status,
        message_type=message_type,
    )


class SentenceAssembler:
    """
    Reassembles multi-part AIVDM messages.

    Type 5 static reports do not fit in one sentence, so they arrive split. Parts
    can interleave with other traffic and can go missing entirely; incomplete
    groups are simply never emitted rather than being padded into something that
    decodes to nonsense.
    """

    def __init__(self) -> None:
        self._pending: dict[str, dict[int, AivdmSentence]] = {}

    def push(self, line: str, timestamp: float = 0.0) -> str | None:
        """Feed one sentence. Returns a complete bit string when a group finishes."""
        sentence = parse_aivdm(line)
        if sentence.fragment_count == 1:
            return sixbit_to_bits(sentence.payload, sentence.fill_bits)

        key = f"{sentence.sequence_id}:{sentence.channel}"
        group = self._pending.setdefault(key, {})
        group[sentence.fragment_number] = sentence
        if len(group) < sentence.fragment_count:
            return None

        ordered = [group[i] for i in sorted(group)]
        self._pending.pop(key, None)
        return "".join(sixbit_to_bits(s.payload, s.fill_bits if s is ordered[-1] else 0)
                       for s in ordered)

    @property
    def incomplete_groups(self) -> int:
        return len(self._pending)


def decode_stream(lines: Iterable[str], timestamps: Sequence[float] | None = None) -> list[Position]:
    """
    Decode a stream of sentences into positions.

    Undecodable sentences are counted and skipped, never guessed at. A feed with
    a high skip rate is telling you something about the feed.
    """
    assembler = SentenceAssembler()
    positions: list[Position] = []
    for index, line in enumerate(lines):
        timestamp = timestamps[index] if timestamps is not None and index < len(timestamps) else 0.0
        try:
            bits = assembler.push(line, timestamp)
        except AisDecodeError:
            continue
        if bits is None:
            continue
        try:
            positions.append(decode_position(bits, timestamp))
        except AisDecodeError:
            continue                       # static/voyage messages land here; not an error
    return positions


# ── geometry ──────────────────────────────────────────────────────────────────

def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in nautical miles."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return float(2 * EARTH_RADIUS_NM * math.asin(min(1.0, math.sqrt(a))))


# ── gap analysis ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Track:
    mmsi: int
    positions: tuple[Position, ...]

    @property
    def duration_hours(self) -> float:
        if len(self.positions) < 2:
            return 0.0
        return (self.positions[-1].timestamp - self.positions[0].timestamp) / 3600.0


@dataclass(frozen=True)
class Gap:
    """A period of silence. A LEAD, not a finding."""

    mmsi: int
    started: float
    ended: float
    distance_nm: float
    last_lat: float
    last_lon: float
    resumed_lat: float
    resumed_lon: float

    @property
    def hours(self) -> float:
        return (self.ended - self.started) / 3600.0

    @property
    def implied_speed_knots(self) -> float:
        return self.distance_nm / self.hours if self.hours > 0 else float("inf")

    def implausible(self, max_speed_knots: float = 25.0) -> bool:
        """
        Faster than the vessel can physically go.

        This is the one signal a coverage hole cannot explain: if the ship
        reappears somewhere it could not have sailed to in the time available,
        either the identity is wrong or one of the two positions is. Worth more
        than gap length alone, which is mostly a map of receiver coverage.
        """
        return self.implied_speed_knots > max_speed_knots

    def assessment(self, max_speed_knots: float = 25.0) -> str:
        if self.implausible(max_speed_knots):
            return (f"reappeared {self.distance_nm:.0f} nm away in {self.hours:.1f} h "
                    f"({self.implied_speed_knots:.0f} kn implied) - check identity spoofing "
                    f"or a bad fix before anything else")
        return (f"{self.hours:.1f} h of silence, {self.distance_nm:.0f} nm covered "
                f"({self.implied_speed_knots:.1f} kn) - consistent with ordinary "
                f"coverage loss; corroborate with imagery before drawing any conclusion")


def find_gaps(track: Track, threshold_hours: float = 6.0) -> list[Gap]:
    """Silences longer than `threshold_hours`. Tune the threshold to the region's coverage."""
    gaps: list[Gap] = []
    ordered = sorted(track.positions, key=lambda p: p.timestamp)
    for before, after in zip(ordered, ordered[1:]):
        if (after.timestamp - before.timestamp) / 3600.0 < threshold_hours:
            continue
        gaps.append(Gap(
            mmsi=track.mmsi, started=before.timestamp, ended=after.timestamp,
            distance_nm=haversine_nm(before.latitude, before.longitude,
                                     after.latitude, after.longitude),
            last_lat=before.latitude, last_lon=before.longitude,
            resumed_lat=after.latitude, resumed_lon=after.longitude,
        ))
    return gaps


@dataclass(frozen=True)
class Rendezvous:
    """Two vessels close together and both nearly stopped. Again: a lead."""

    mmsi_a: int
    mmsi_b: int
    started: float
    ended: float
    min_distance_nm: float
    latitude: float
    longitude: float

    @property
    def hours(self) -> float:
        return (self.ended - self.started) / 3600.0

    def __str__(self) -> str:
        return (f"{self.mmsi_a} and {self.mmsi_b}: {self.hours:.1f} h within "
                f"{self.min_distance_nm:.2f} nm at ({self.latitude:.3f}, {self.longitude:.3f})")


def find_rendezvous(tracks: Sequence[Track], max_distance_nm: float = 0.5,
                    min_hours: float = 2.0, max_speed_knots: float = 1.0,
                    tolerance_seconds: float = 900.0) -> list[Rendezvous]:
    """
    Pairs of vessels that were close and slow at the same time.

    That pattern is consistent with a ship-to-ship transfer. It is also
    consistent with anchoring in the same anchorage, waiting for a berth, a pilot
    boarding, bunkering, or a crew transfer - all of which are ordinary and legal
    and vastly more common. The output ranks pairs for a human to look at.
    """
    out: list[Rendezvous] = []
    for i, first in enumerate(tracks):
        for second in tracks[i + 1:]:
            contacts = _close_contacts(first, second, max_distance_nm,
                                       max_speed_knots, tolerance_seconds)
            if not contacts:
                continue
            span = (contacts[-1][0] - contacts[0][0]) / 3600.0
            if span < min_hours:
                continue
            closest = min(contacts, key=lambda c: c[1])
            out.append(Rendezvous(first.mmsi, second.mmsi, contacts[0][0], contacts[-1][0],
                                  closest[1], closest[2], closest[3]))
    return sorted(out, key=lambda r: r.hours, reverse=True)


def _close_contacts(a: Track, b: Track, max_distance_nm: float,
                    max_speed_knots: float, tolerance_seconds: float):
    contacts = []
    for position in a.positions:
        partner = _nearest_in_time(b.positions, position.timestamp, tolerance_seconds)
        if partner is None:
            continue
        if (position.sog_knots or 0.0) > max_speed_knots or (partner.sog_knots or 0.0) > max_speed_knots:
            continue
        distance = haversine_nm(position.latitude, position.longitude,
                                partner.latitude, partner.longitude)
        if distance <= max_distance_nm:
            contacts.append((position.timestamp, distance, position.latitude, position.longitude))
    return sorted(contacts)


def _nearest_in_time(positions: Sequence[Position], when: float, tolerance: float):
    best, best_delta = None, tolerance
    for position in positions:
        delta = abs(position.timestamp - when)
        if delta <= best_delta:
            best, best_delta = position, delta
    return best


def build_tracks(positions: Sequence[Position]) -> list[Track]:
    grouped: dict[int, list[Position]] = {}
    for position in positions:
        if position.position_valid:
            grouped.setdefault(position.mmsi, []).append(position)
    return [Track(mmsi, tuple(sorted(ps, key=lambda p: p.timestamp)))
            for mmsi, ps in sorted(grouped.items())]


# ── synthetic encoder, for tests and demos ────────────────────────────────────

def encode_position(mmsi: int, latitude: float, longitude: float, sog: float = 0.0,
                    cog: float = 0.0, message_type: int = 1) -> str:
    """Build a valid type 1 AIVDM sentence. Used to test the decoder round-trip."""
    bits = (format(message_type, "06b") + "00" + format(mmsi, "030b") + format(0, "04b")
            + format(0, "08b") + format(int(round(sog * 10)), "010b") + "0"
            + format(int(round(longitude * 600_000)) & 0xFFFFFFF, "028b")
            + format(int(round(latitude * 600_000)) & 0x7FFFFFF, "027b")
            + format(int(round(cog * 10)), "012b") + format(511, "09b")
            + format(60, "06b") + "0" * 8)
    payload, fill = bits_to_sixbit(bits)
    body = f"AIVDM,1,1,,A,{payload},{fill}"
    return f"!{body}*{nmea_checksum(body)}"


if __name__ == "__main__":
    print("STARK MK18 - AIS decoding and gap analysis\n")

    sentence = encode_position(mmsi=477_123_456, latitude=1.2650, longitude=103.8200, sog=12.4)
    print(f"encoded:  {sentence}")
    decoded = decode_position(sixbit_to_bits(parse_aivdm(sentence).payload,
                                             parse_aivdm(sentence).fill_bits))
    print(f"decoded:  MMSI {decoded.mmsi} at ({decoded.latitude:.4f}, {decoded.longitude:.4f}) "
          f"doing {decoded.sog_knots} kn")
    print(f"checksum: {'valid' if parse_aivdm(sentence).checksum_ok else 'INVALID'}\n")

    hour = 3600.0
    laden = [Position(477_111_111, t * hour, 1.20 + 0.01 * t, 103.80 + 0.01 * t,
                      11.0, 90.0, None, 0, 1) for t in range(0, 6)]
    # ...then eleven hours of silence, reappearing 40 nm on
    laden += [Position(477_111_111, t * hour, 1.90, 104.35, 10.0, 90.0, None, 0, 1)
              for t in range(17, 22)]

    gaps = find_gaps(Track(477_111_111, tuple(laden)), threshold_hours=6.0)
    print(f"gap analysis on one vessel: {len(gaps)} gap(s) over "
          f"{Track(477_111_111, tuple(laden)).duration_hours:.0f} hours")
    for gap in gaps:
        print(f"  MMSI {gap.mmsi}: {gap.assessment()}")

    stationary_a = [Position(477_222_222, t * hour, 3.5000, 105.2000, 0.2, 0, None, 1, 1)
                    for t in range(10, 18)]
    stationary_b = [Position(636_333_333, t * hour, 3.5010, 105.2008, 0.3, 0, None, 1, 1)
                    for t in range(10, 18)]
    meetings = find_rendezvous([Track(477_222_222, tuple(stationary_a)),
                                Track(636_333_333, tuple(stationary_b))])
    print(f"\nrendezvous screen: {len(meetings)} pair(s) close and slow together")
    for meeting in meetings:
        print(f"  {meeting}")

    print("\nWhat these flags mean: look here next. Nothing more.")
    print("A gap is usually a receiver coverage hole. Two ships close and slow are usually")
    print("two ships in an anchorage. Corroborate with imagery, port calls and draught")
    print("before anyone says a word about a named vessel - and note that the implausible")
    print("-speed test is the only one here a coverage hole cannot explain away.")

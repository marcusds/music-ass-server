"""Tests for the rate at which stream output is handed to a player."""

from __future__ import annotations

from typing import cast

import pytest
from music_assistant_models.enums import ContentType, MediaType, StreamType
from music_assistant_models.media_items import AudioFormat
from music_assistant_models.streamdetails import StreamDetails

from music_assistant.constants import CONF_ENTRY_CROSSFADE_DURATION
from music_assistant.controllers.streams.constants import (
    FLOW_STREAM_LEAD_OUT_SECONDS,
    PacingProfile,
    flow_pacing_profile,
    output_pacing_args,
)

# A track of ordinary length from a source that delivers just-in-time, at the low end
# of what the Spotify Soloist backend delivered when measured (1.07x to 1.1x).
TRACK_SECONDS = 200
SLOW_SOURCE_RATE = 1.075
# The longest such a source took to have a track ready: up to 1.7 seconds to start,
# plus the first second of audio its stream waits for.
SLOW_SOURCE_START_SECONDS = 2.7


def _value(args: list[str], key: str) -> float:
    return float(args[args.index(key) + 1])


def test_pacing_stays_ahead_of_playback() -> None:
    """
    At or below playback rate a player would underrun as soon as its burst ran out.

    The ceiling may be lowered per player, but never to realtime or slower.
    """
    assert _value(output_pacing_args(), "-readrate") > 1.0
    assert _value(output_pacing_args(PacingProfile.NEAR_REALTIME), "-readrate") > 1.0
    assert _value(output_pacing_args(PacingProfile.LOW_LATENCY), "-readrate") > 1.0


def test_the_default_burst_covers_a_gapless_players_opening_chunk() -> None:
    """A player that holds a whole opening chunk before it plays gapless must get one."""
    assert _value(output_pacing_args(), "-readrate_initial_burst") >= 10


def test_the_near_realtime_profile_leaves_room_for_a_source_to_bank_ahead() -> None:
    """
    A just-in-time source delivers ~1.1x at best, and its bank is what a crossfade mixes.

    Drained at the default pace that bank never grows, so the profile has to stay
    below it and leave real margin under what the source can deliver. The burst
    comes out of that same bank, so it stays small too.
    """
    readrate = _value(output_pacing_args(PacingProfile.NEAR_REALTIME), "-readrate")
    assert readrate < _value(output_pacing_args(), "-readrate")
    # 1.1 is the fill rate, so anything close to it banks nothing
    assert readrate <= 1.05
    assert _value(output_pacing_args(PacingProfile.NEAR_REALTIME), "-readrate_initial_burst") <= 5


def test_the_near_realtime_pacing_is_the_measured_one() -> None:
    """
    The pace and the burst were picked from measurements on real speakers.

    Another value needs its own measurements: a higher pace takes the crossfades of a
    slow source away, a lower one or a smaller burst takes the player's cushion.
    """
    assert output_pacing_args(PacingProfile.NEAR_REALTIME) == [
        "-readrate",
        "1.01",
        "-readrate_initial_burst",
        "3",
    ]


def test_a_track_of_a_slow_source_earns_the_room_for_a_fade() -> None:
    """
    A source fills its next track as soon as the current one is in.

    The room for a fade is how much longer the stream of a track takes than its fill,
    and a fade of the default length needs that room at the end of the first track
    already. A stream per track is the tight case, as its burst comes out of that room
    with every track, where the flow stream opens with one burst for a whole session.
    """
    args = output_pacing_args(PacingProfile.NEAR_REALTIME)
    stream_seconds = (TRACK_SECONDS - _value(args, "-readrate_initial_burst")) / _value(
        args, "-readrate"
    )
    fill_seconds = TRACK_SECONDS / SLOW_SOURCE_RATE
    default_fade_seconds = cast("int", CONF_ENTRY_CROSSFADE_DURATION.default_value)
    assert stream_seconds - fill_seconds >= default_fade_seconds


def test_the_opening_burst_outlasts_a_source_starting_up() -> None:
    """
    A track change without a fade holds the stream back until the next track's source delivers.

    At the first track change of any stream the burst is all a player is sure to
    hold, however short that first track is.
    """
    burst = _value(output_pacing_args(PacingProfile.NEAR_REALTIME), "-readrate_initial_burst")
    assert burst > SLOW_SOURCE_START_SECONDS


def test_the_low_latency_burst_stays_under_a_second() -> None:
    """A live source's burst is listening delay: the player buffers it ahead of real time."""
    assert _value(output_pacing_args(PacingProfile.LOW_LATENCY), "-readrate_initial_burst") <= 1


def _details(
    *, is_realtime: bool = False, duration: int | None = 200, seek_position: int = 0
) -> StreamDetails:
    return StreamDetails(
        provider="builtin",
        item_id="item-1",
        audio_format=AudioFormat(content_type=ContentType.FLAC),
        media_type=MediaType.TRACK,
        stream_type=StreamType.HTTP,
        path="http://example.com/audio.flac",
        duration=duration,
        seek_position=seek_position,
        is_realtime=is_realtime,
    )


def test_a_buffered_flow_keeps_the_realtime_pace() -> None:
    """
    Only the opening burst of a flow differs by its opening item, never its pace.

    A just-in-time source later in the same flow banks against that pace for its
    crossfade, whatever the session opened on.
    """
    buffered = output_pacing_args(PacingProfile.BUFFERED_FLOW)
    near_realtime = output_pacing_args(PacingProfile.NEAR_REALTIME)
    assert _value(buffered, "-readrate") == _value(near_realtime, "-readrate")


def test_a_buffered_flow_opens_with_a_lead_for_its_track_boundaries() -> None:
    """
    A flow resolves every next item inline, and a slow boundary plays out of the player's lead.

    That lead should outlast a next item waiting on a source slot for a few seconds, and
    stay a small fraction of what a single track gets, which a Chromecast takes too.
    """
    burst = _value(output_pacing_args(PacingProfile.BUFFERED_FLOW), "-readrate_initial_burst")
    assert burst >= 2 * _value(
        output_pacing_args(PacingProfile.NEAR_REALTIME), "-readrate_initial_burst"
    )
    assert burst <= _value(output_pacing_args(), "-readrate_initial_burst") / 4


def test_the_buffered_flow_burst_stays_near_the_lead_out() -> None:
    """A queue that ends right after the burst still plays out nearly all of it."""
    burst = _value(output_pacing_args(PacingProfile.BUFFERED_FLOW), "-readrate_initial_burst")
    assert burst <= FLOW_STREAM_LEAD_OUT_SECONDS + 2


@pytest.mark.parametrize(
    ("details", "profile"),
    [
        (_details(), PacingProfile.BUFFERED_FLOW),
        (_details(is_realtime=True), PacingProfile.NEAR_REALTIME),
        (None, PacingProfile.NEAR_REALTIME),
        (_details(duration=None), PacingProfile.NEAR_REALTIME),
        (_details(duration=8), PacingProfile.NEAR_REALTIME),
        (_details(duration=200, seek_position=195), PacingProfile.NEAR_REALTIME),
        (_details(duration=200, seek_position=150), PacingProfile.BUFFERED_FLOW),
    ],
    ids=[
        "buffered",
        "realtime",
        "no_details",
        "unknown_duration",
        "shorter_than_the_burst",
        "seeked_into_the_burst",
        "seeked_with_room",
    ],
)
def test_flow_pacing_takes_its_burst_only_from_the_opening_item(
    details: StreamDetails | None, profile: PacingProfile
) -> None:
    """
    The larger burst is only taken when the opening item can hand all of it over itself.

    A realtime item cannot hand it over without draining what it banks for its
    crossfade, and an item that ends inside it would hand the rest out of the item
    after it, which may be just such a source.
    """
    assert flow_pacing_profile(details) == profile

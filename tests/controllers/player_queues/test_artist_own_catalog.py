"""
Tests for playing an artist's own catalog.

Playing an artist with the all tracks selection plays the tracks the artist is the first artist
of, or an album artist of, and leaves out covers (karaoke, tribute and "originally performed by"
releases), the songs of others it only features on, and the provider albums of other artists.
"""

from __future__ import annotations

from functools import partial
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from music_assistant_models.enums import AlbumType, ExternalID, MediaType
from music_assistant_models.media_items import (
    Album,
    Artist,
    ItemMapping,
    ProviderMapping,
    Track,
)
from music_assistant_models.unique_list import UniqueList

from music_assistant.controllers.player_queues.constants import ArtistOrder
from music_assistant.controllers.player_queues.media_resolver import MediaResolver
from music_assistant.helpers.compare import (
    dedupe_track_recordings,
    is_artist_own_album,
    is_artist_own_track,
    is_cover_release,
)


def _artist(name: str, item_id: str | None = None, provider: str = "spotify") -> Artist:
    """Build an artist as a provider lists it."""
    item_id = item_id or name.lower().replace(" ", "-")
    return Artist(
        item_id=item_id,
        provider=provider,
        name=name,
        provider_mappings={
            ProviderMapping(item_id=item_id, provider_domain=provider, provider_instance=provider)
        },
    )


BEATLES = _artist("The Beatles", "beatles")


def _track(
    name: str,
    artists: list[Artist],
    version: str = "",
    album: Album | None = None,
    duration: int = 0,
    provider: str = "spotify",
) -> Track:
    """Build a track credited to the given artists, in order."""
    return Track(
        item_id=f"{name}-{version}-{artists[0].name if artists else ''}",
        provider=provider,
        name=name,
        version=version,
        duration=duration,
        artists=UniqueList(artists),
        album=album,
        provider_mappings=set(),
    )


def _album(name: str, artists: list[Artist], version: str = "") -> Album:
    """Build an album credited to the given artists."""
    return Album(
        item_id=name,
        provider="spotify",
        name=name,
        version=version,
        album_type=AlbumType.ALBUM,
        artists=UniqueList(artists),
        provider_mappings=set(),
    )


@pytest.mark.parametrize(
    ("name", "version"),
    [
        ("Yesterday", "Karaoke Version"),
        ("Yesterday (Originally Performed by The Beatles)", ""),
        ("Hey Jude - Made Famous by The Beatles", ""),
        ("Let It Be [In the Style of The Beatles]", ""),
        ("Help!", "Tribute"),
        ("Something", "Backing Track"),
    ],
)
def test_cover_releases_are_recognized(name: str, version: str) -> None:
    """A karaoke, tribute or "originally performed by" release is a cover."""
    assert is_cover_release(name, version)


@pytest.mark.parametrize(
    ("name", "version"),
    [
        ("Tribute", ""),
        ("Twist and Shout", "Live"),
        ("Come Together", "Remastered 2009"),
        ("Revolution 9", ""),
        ("Because", "Instrumental"),
        ("Cover Me", ""),
    ],
)
def test_own_versions_are_no_covers(name: str, version: str) -> None:
    """A live take, remaster, instrumental or a song named like a marker is not a cover."""
    assert not is_cover_release(name, version)


def test_primary_artist_track_is_own() -> None:
    """A track the artist is the first artist of is in its catalog, with other artists after."""
    track = _track("Get Back", [BEATLES, _artist("Billy Preston")])

    assert is_artist_own_track(track, [BEATLES])


def test_featured_appearance_is_not_own() -> None:
    """A track of another artist that the artist only features on is left out."""
    track = _track("My Bonnie", [_artist("Tony Sheridan"), BEATLES])

    assert not is_artist_own_track(track, [BEATLES])


def test_cover_by_another_artist_is_not_own() -> None:
    """A cover by another artist is left out."""
    track = _track("With a Little Help from My Friends", [_artist("Joe Cocker")])

    assert not is_artist_own_track(track, [BEATLES])


def test_karaoke_credited_to_the_artist_is_not_own() -> None:
    """A karaoke release is left out, even when a provider credits the artist."""
    track = _track("Yesterday", [BEATLES], version="Karaoke Version")

    assert not is_artist_own_track(track, [BEATLES])


def test_album_artist_track_is_own() -> None:
    """A track on an album of the artist is in its catalog, whoever is its first artist."""
    album = _album("Abbey Road", [BEATLES])
    track = _track("Something", [_artist("George Harrison")], album=album)

    assert is_artist_own_track(track, [BEATLES])


def test_track_matches_through_the_library_artist() -> None:
    """A provider track matches the library item of the artist through its mapping."""
    library_artist = Artist(
        item_id="12",
        provider="library",
        name="Beatles, The",
        provider_mappings={
            ProviderMapping(
                item_id="beatles", provider_domain="spotify", provider_instance="spotify"
            )
        },
    )
    provider_credit = ItemMapping(
        media_type=MediaType.ARTIST, item_id="beatles", provider="spotify", name="The Beatles"
    )
    track = _track("Help!", [cast("Artist", provider_credit)])

    assert is_artist_own_track(track, [library_artist])


def test_artist_with_another_external_id_is_not_own() -> None:
    """A namesake artist with another MusicBrainz id is a different artist."""
    beatles = _artist("The Beatles")
    beatles.external_ids = {(ExternalID.MB_ARTIST, "b10bbbfc-cf9e-42e0-be17-e2c3e1d2600d")}
    namesake = _artist("The Beatles", "namesake", provider="other")
    namesake.external_ids = {(ExternalID.MB_ARTIST, "00000000-0000-0000-0000-000000000000")}

    assert not is_artist_own_track(_track("Song", [namesake]), [beatles])


def test_library_track_counts_any_credited_artist() -> None:
    """The library keeps no artist order, so a library track counts any of its artists."""
    library_beatles = Artist(
        item_id="1", provider="library", name="The Beatles", provider_mappings=set()
    )
    other = Artist(item_id="2", provider="library", name="Billy Preston", provider_mappings=set())
    track = _track("Get Back", [other, library_beatles], provider="library")

    assert is_artist_own_track(track, [library_beatles])


def test_track_without_artists_is_own() -> None:
    """A track that lists no artists can't be told apart, so it stays."""
    assert is_artist_own_track(_track("Unknown", []), [BEATLES])


def test_tribute_album_by_other_artists_is_not_own() -> None:
    """A tribute compilation by other artists is not an album of the artist."""
    assert not is_artist_own_album(
        _album("A Tribute to the Beatles", [_artist("Various Artists")]), [BEATLES]
    )
    assert not is_artist_own_album(_album("Abbey Road", [BEATLES], "Karaoke Edition"), [BEATLES])
    assert is_artist_own_album(_album("Live at the BBC", [BEATLES]), [BEATLES])
    assert is_artist_own_album(_album("Unknown", []), [BEATLES])


def test_dedupe_keeps_different_songs_sharing_a_title() -> None:
    """Two songs that share a title are both kept, while repeats of a recording are dropped."""
    tracks = [
        _track("Intro", [BEATLES], duration=60),
        _track("Intro", [BEATLES], duration=180),
        _track("Intro", [BEATLES], duration=62),
        _track("Help!", [BEATLES], duration=138),
        _track("help!", [BEATLES], duration=0),
        _track("Help!", [BEATLES], version="Live", duration=150),
    ]

    result = dedupe_track_recordings(tracks)

    assert [(track.name, track.version, track.duration) for track in result] == [
        ("Intro", "", 60),
        ("Intro", "", 180),
        ("Help!", "", 138),
        ("Help!", "Live", 150),
    ]


def _fake_resolver(library: list[Track], provider: list[Track]) -> MagicMock:
    """Create a mock standing in for the media resolver, playing the all tracks selection."""
    fake = MagicMock()
    fake.mass.config.get_raw_core_config_value = MagicMock(return_value="all_tracks")
    fake._resolve_library_artist = AsyncMock(return_value=None)
    fake.get_artist_order = MagicMock(return_value=ArtistOrder.SHUFFLED_TRACKS)
    fake._album_order_applies = MagicMock(return_value=False)
    fake._library_artist_tracks = AsyncMock(return_value=library)
    fake._provider_artist_tracks = AsyncMock(return_value=provider)
    fake.resolve_artist_tracks = partial(MediaResolver.resolve_artist_tracks, fake)
    fake._shuffled_artist_tracks = partial(MediaResolver._shuffled_artist_tracks, fake)
    return fake


async def test_all_tracks_plays_the_own_catalog() -> None:
    """All tracks plays the artist's songs, without covers and songs of others it features on."""
    fake = _fake_resolver(
        library=[],
        provider=[
            _track("Help!", [BEATLES]),
            _track("Twist and Shout", [BEATLES], version="Live"),
            _track(
                "Something", [_artist("George Harrison")], album=_album("Abbey Road", [BEATLES])
            ),
            _track("Yesterday", [_artist("Sing King")], version="Karaoke Version"),
            _track("Yesterday (Originally Performed by The Beatles)", [BEATLES]),
            _track("My Bonnie", [_artist("Tony Sheridan"), BEATLES]),
            _track("Come Together", [_artist("Aerosmith")]),
        ],
    )

    result = await MediaResolver.get_artist_tracks(cast("MediaResolver", fake), BEATLES)

    assert sorted(track.name for track in result) == ["Help!", "Something", "Twist and Shout"]


async def test_artist_without_own_tracks_plays_its_appearances() -> None:
    """An artist that only features on songs of others still plays those, but no covers."""
    session_player = _artist("Session Player")
    fake = _fake_resolver(
        library=[],
        provider=[
            _track("Hit", [_artist("Star"), session_player]),
            _track("Hit", [_artist("Karaoke Kings")], version="Karaoke"),
        ],
    )

    result = await MediaResolver.get_artist_tracks(cast("MediaResolver", fake), session_player)

    assert [(track.name, track.version) for track in result] == [("Hit", "")]

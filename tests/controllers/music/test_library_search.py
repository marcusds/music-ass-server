"""Tests for the word matching and relevance ranking of library searches."""

from __future__ import annotations

from uuid import uuid4

import pytest
from music_assistant_models.enums import AlbumType, SortField
from music_assistant_models.media_items import Album, Artist, ProviderMapping, Track
from music_assistant_models.unique_list import UniqueList

from music_assistant.mass import MusicAssistant

pytestmark = pytest.mark.asyncio


@pytest.fixture(scope="module")
async def search_mass(music_mass_module: MusicAssistant) -> MusicAssistant:
    """Return a module-scoped database-only instance with a small seeded library."""
    await _seed_library(music_mass_module)
    return music_mass_module


def _mapping() -> ProviderMapping:
    """Create an in-library provider mapping with a unique provider item id."""
    return ProviderMapping(
        item_id=uuid4().hex,
        provider_domain="prov_a",
        provider_instance="prov_a_inst",
        in_library=True,
    )


async def _seed_library(mass: MusicAssistant) -> None:
    """Seed a few artists, albums and tracks with overlapping names."""
    artists: dict[str, Artist] = {}
    for name in ("The Beatles", "Pink Floyd", "Joni Mitchell", "Abbey Road Band"):
        artists[name] = await mass.music.artists.add_item_to_library(
            Artist(item_id="0", provider="library", name=name, provider_mappings={_mapping()})
        )
    albums: dict[str, Album] = {}
    for name, artist in (
        ("Abbey Road", "The Beatles"),
        ("Abbey Road Sessions", "The Beatles"),
        ("Road To Abbey", "Pink Floyd"),
        ("Live", "Abbey Road Band"),
        ("Help!", "The Beatles"),
        ("Court and Spark", "Joni Mitchell"),
    ):
        albums[name] = await mass.music.albums.add_item_to_library(
            Album(
                item_id="0",
                provider="library",
                name=name,
                album_type=AlbumType.ALBUM,
                provider_mappings={_mapping()},
                artists=UniqueList([artists[artist]]),
            )
        )
    for idx, (name, artist, album) in enumerate(
        (
            ("Come Together", "The Beatles", "Abbey Road"),
            ("Something", "The Beatles", "Abbey Road"),
            ("Cry For Help", "Pink Floyd", "Road To Abbey"),
            ("Helpless", "Pink Floyd", "Road To Abbey"),
            ("Help!", "The Beatles", "Help!"),
            ("Help Me", "Joni Mitchell", "Court and Spark"),
        ),
        start=1,
    ):
        await mass.music.tracks.add_item_to_library(
            Track(
                item_id="0",
                provider="library",
                name=name,
                provider_mappings={_mapping()},
                artists=UniqueList([artists[artist]]),
                album=albums[album],
                disc_number=1,
                track_number=idx,
            )
        )
    await mass.music.database.commit()


async def test_words_match_in_any_order(search_mass: MusicAssistant) -> None:
    """A search finds an item when its words are typed in another order."""
    albums = await search_mass.music.albums.library_items(search="road abbey")
    names = [album.name for album in albums]
    assert set(names[:3]) == {"Abbey Road", "Abbey Road Sessions", "Road To Abbey"}
    # the artist "Abbey Road Band" matches both words too, but not in the album name
    assert names[3:] == ["Live"]
    # the generic listing query of all media types matches the words too
    albums = await search_mass.music.albums.get_library_items_by_query(search="sessions abbey")
    assert [album.name for album in albums] == ["Abbey Road Sessions"]


async def test_words_spread_over_artist_and_title(search_mass: MusicAssistant) -> None:
    """A search can combine words of the artist name and of the item name."""
    albums = await search_mass.music.albums.library_items(search="beatles abbey road")
    assert [album.name for album in albums] == ["Abbey Road", "Abbey Road Sessions"]
    tracks = await search_mass.music.tracks.library_items(search="together beatles")
    assert [track.name for track in tracks] == ["Come Together"]
    # artist words alone match all items of that artist
    tracks = await search_mass.music.tracks.library_items(search="pink floyd")
    assert {track.name for track in tracks} == {"Cry For Help", "Helpless"}


async def test_short_words_are_matched(search_mass: MusicAssistant) -> None:
    """Words too short for the trigram index still narrow the result."""
    tracks = await search_mass.music.tracks.library_items(search="help me")
    assert [track.name for track in tracks] == ["Help Me"]


async def test_results_are_ranked_on_relevance(search_mass: MusicAssistant) -> None:
    """Exact, prefix, substring, all-words and other matches come in that order."""
    tracks = await search_mass.music.tracks.library_items(search="help")
    assert [track.name for track in tracks] == ["Help!", "Helpless", "Help Me", "Cry For Help"]
    albums = await search_mass.music.albums.library_items(search="abbey road")
    assert [album.name for album in albums] == [
        "Abbey Road",
        "Abbey Road Sessions",
        "Road To Abbey",
        "Live",
    ]
    # the library leg of the global search uses the same ranking
    tracks = await search_mass.music.tracks.search("help", "library")
    assert [track.name for track in tracks][:2] == ["Help!", "Helpless"]


async def test_explicit_sort_is_kept(search_mass: MusicAssistant) -> None:
    """A sort requested by the caller wins over the relevance ranking."""
    tracks = await search_mass.music.tracks.library_items(search="help", sort_field=SortField.NAME)
    assert [track.name for track in tracks] == ["Cry For Help", "Help!", "Helpless", "Help Me"]
    tracks = await search_mass.music.tracks.library_items(
        search="pink floyd", sort_field=SortField.ARTIST_NAME
    )
    assert {track.name for track in tracks} == {"Cry For Help", "Helpless"}


async def test_ranked_pages_are_stable(search_mass: MusicAssistant) -> None:
    """Paging a ranked search returns every match once, in the order of the full listing."""
    full = await search_mass.music.albums.library_items(search="abbey road")
    paged = [
        album
        for offset in range(len(full))
        for album in await search_mass.music.albums.get_library_items_by_query(
            search="abbey road",
            limit=1,
            offset=offset,
            sort_field=SortField.SORT_NAME,
            rank_search="abbey road",
        )
    ]
    single = await search_mass.music.albums.get_library_items_by_query(
        search="abbey road", sort_field=SortField.SORT_NAME, rank_search="abbey road"
    )
    assert [album.item_id for album in paged] == [album.item_id for album in single]


async def test_count_matches_listing(search_mass: MusicAssistant) -> None:
    """The count of a word search agrees with its listing."""
    items = await search_mass.music.albums.get_library_items_by_query(search="road abbey")
    count = await search_mass.music.albums.get_library_items_count_by_query(search="road abbey")
    assert count == len(items) == 3


async def test_artist_fallback_keeps_limit_and_filters(search_mass: MusicAssistant) -> None:
    """Tracks appended for an artist name match respect the limit and the played filter."""
    tracks = await search_mass.music.tracks.library_items(search="beatles", limit=2)
    assert len(tracks) == 2
    assert not await search_mass.music.tracks.library_items(search="beatles", played_only=True)

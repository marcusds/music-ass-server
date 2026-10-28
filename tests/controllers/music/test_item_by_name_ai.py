"""Tests for the AI assisted lookup of music/item_by_name."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, NamedTuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from music_assistant_models.enums import MediaType
from music_assistant_models.media_items import (
    Album,
    Artist,
    ItemMapping,
    SearchResults,
    Track,
    UniqueList,
)

from music_assistant.controllers.music import MusicController, ai_lookup
from music_assistant.controllers.music.ai_lookup import (
    AILookupTerms,
    build_ai_lookup_prompt,
    parse_ai_lookup_response,
)
from music_assistant.controllers.music.constants import (
    CONF_AI_ITEM_LOOKUP,
    CONF_AI_ITEM_LOOKUP_ENGINE,
)
from music_assistant.models.plugin import AIEngine, PluginProvider

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from music_assistant.mass import MusicAssistant

AI_REPLY = '{"media_type": "track", "name": "Yesterday", "artist": "The Beatles", "album": null}'


def _track(name: str, artist: str = "The Beatles") -> Track:
    """Create a provider track by the given artist."""
    return Track(
        item_id=name,
        provider="test",
        name=name,
        provider_mappings=set(),
        artists=UniqueList(
            [ItemMapping(item_id=artist, provider="test", name=artist, media_type=MediaType.ARTIST)]
        ),
    )


def _plugin(reply: str | Exception | None = AI_REPLY) -> MagicMock:
    """Create a plugin exposing one AI engine answering with the given reply."""
    plugin = MagicMock(spec=PluginProvider)
    plugin.instance_id = "ai_plugin"
    plugin.name = "AI plugin"
    plugin.get_ai_engines = AsyncMock(
        return_value=[AIEngine(id="model", name="Model", provider=plugin)]
    )
    plugin.ai_query = AsyncMock(
        side_effect=reply if isinstance(reply, Exception) else None, return_value=reply
    )
    return plugin


async def _search(search_query: str, **_kwargs: object) -> SearchResults:
    """Return the right track only for a correctly spelled query."""
    if "yesterday" in search_query.lower():
        return SearchResults(tracks=[_track("Yesterday")])
    return SearchResults(tracks=[_track("Yellow Submarine")])


class Lookup(NamedTuple):
    """A music controller with a fake library and a fake global search."""

    music: MusicController
    search: AsyncMock
    library_items: AsyncMock


@pytest.fixture
async def lookup(mass_minimal: MusicAssistant) -> AsyncGenerator[Lookup]:
    """Return a music controller with an empty library and a fake global search."""
    controller = MusicController(mass_minimal)
    mass_minimal.music = controller
    media_controller = MagicMock()
    media_controller.library_items = AsyncMock(return_value=[])
    search = AsyncMock(side_effect=_search)
    with (
        patch.object(controller, "get_controller", return_value=media_controller),
        patch.object(controller, "search", search),
    ):
        yield Lookup(controller, search, media_controller.library_items)


def _serve(mass: MusicAssistant, *plugins: MagicMock) -> None:
    """Let the given plugins be the AI engine providers of the instance."""
    mass.get_providers_supporting_feature = MagicMock(  # type: ignore[method-assign]
        return_value=list(plugins)
    )


async def test_typo_resolves_through_ai_engine(lookup: Lookup) -> None:
    """A misspelled request is corrected by the AI engine and found on the second search."""
    plugin = _plugin()
    _serve(lookup.music.mass, plugin)

    item = await lookup.music.get_item_by_name("Yesturday", artist="beatls")

    assert item is not None
    assert item.name == "Yesterday"
    plugin.ai_query.assert_awaited_once()
    prompt = plugin.ai_query.await_args.args[0]
    assert '{"name":"Yesturday","artist":"beatls","album":null}' in prompt
    assert plugin.ai_query.await_args.kwargs == {"engine_id": "model"}
    # the first engine was adopted as the core selection
    assert (
        lookup.music.mass.config.get_raw_core_config_value(
            lookup.music.domain, CONF_AI_ITEM_LOOKUP_ENGINE
        )
        == "ai_plugin/model"
    )


async def test_close_match_does_not_ask_the_engine(lookup: Lookup) -> None:
    """A request that already finds a matching item never reaches the AI engine."""
    plugin = _plugin()
    _serve(lookup.music.mass, plugin)

    item = await lookup.music.get_item_by_name("Yesterday")

    assert item is not None
    assert item.name == "Yesterday"
    plugin.ai_query.assert_not_awaited()


async def test_library_exact_match_does_not_ask_the_engine(lookup: Lookup) -> None:
    """An exact library match is returned without search or AI."""
    plugin = _plugin()
    _serve(lookup.music.mass, plugin)
    library_track = _track("Yesterday")
    lookup.library_items.return_value = [library_track]

    assert await lookup.music.get_item_by_name("yesterday") is library_track
    plugin.ai_query.assert_not_awaited()
    lookup.search.assert_not_awaited()


@pytest.mark.parametrize(
    "reply",
    [
        RuntimeError("engine offline"),
        "Sure! Here is what you asked for.",
        '["Yesterday"]',
        '{"name": ""}',
        None,
    ],
)
async def test_engine_failure_falls_back(lookup: Lookup, reply: str | Exception | None) -> None:
    """An engine error or unusable reply leaves the result of the plain lookup."""
    _serve(lookup.music.mass, _plugin(reply))

    item = await lookup.music.get_item_by_name("Yesturday", artist="beatls")

    assert item is not None
    assert item.name == "Yellow Submarine"
    lookup.search.assert_awaited_once()


async def test_engine_timeout_falls_back(lookup: Lookup) -> None:
    """A slow engine is cut off and the plain lookup result is returned."""
    plugin = _plugin()

    async def _slow(*_args: object, **_kwargs: object) -> str:
        await asyncio.sleep(10)
        return AI_REPLY

    plugin.ai_query.side_effect = _slow
    _serve(lookup.music.mass, plugin)

    with patch.object(ai_lookup, "AI_LOOKUP_TIMEOUT_SECONDS", 0.05):
        async with asyncio.timeout(2):
            item = await lookup.music.get_item_by_name("Yesturday", artist="beatls")

    assert item is not None
    assert item.name == "Yellow Submarine"


async def test_without_engine_behaviour_is_unchanged(lookup: Lookup) -> None:
    """Without an AI engine the first search result is returned, as before."""
    _serve(lookup.music.mass)

    item = await lookup.music.get_item_by_name("Yesturday", artist="beatls")

    assert item is not None
    assert item.name == "Yellow Submarine"
    lookup.search.assert_awaited_once()


async def test_disabled_option_skips_the_engine(lookup: Lookup) -> None:
    """The user can turn the AI lookup off."""
    plugin = _plugin()
    _serve(lookup.music.mass, plugin)
    lookup.music.mass.config.set_raw_core_config_value(
        lookup.music.domain, CONF_AI_ITEM_LOOKUP, False
    )

    item = await lookup.music.get_item_by_name("Yesturday", artist="beatls")

    assert item is not None
    assert item.name == "Yellow Submarine"
    plugin.ai_query.assert_not_awaited()


async def test_nothing_found_returns_unconfident_ai_result(lookup: Lookup) -> None:
    """When the plain lookup finds nothing, the AI lookup result is used even if loose."""
    plugin = _plugin(
        '{"media_type": "album", "name": "Abbey Road", "artist": "The Beatles", "album": null}'
    )
    _serve(lookup.music.mass, plugin)
    album = Album(
        item_id="abbey", provider="test", name="Abbey Road (Remastered)", provider_mappings=set()
    )

    async def _album_search(search_query: str, **kwargs: object) -> SearchResults:
        if "abbey road" in search_query.lower():
            assert kwargs["media_types"] == [MediaType.ALBUM]
            return SearchResults(albums=[album])
        return SearchResults()

    lookup.search.side_effect = _album_search

    assert await lookup.music.get_item_by_name("something by the beatls from abby road") is album


async def test_media_type_restricts_the_engine(lookup: Lookup) -> None:
    """A requested media type is the only one offered to and accepted from the engine."""
    plugin = _plugin('{"media_type": "track", "name": "The Beatles"}')
    _serve(lookup.music.mass, plugin)
    artist = Artist(item_id="beatles", provider="test", name="The Beatles", provider_mappings=set())

    async def _artist_search(search_query: str, **kwargs: object) -> SearchResults:
        assert kwargs["media_types"] == [MediaType.ARTIST]
        if search_query == "The Beatles":
            return SearchResults(artists=[artist])
        return SearchResults()

    lookup.search.side_effect = _artist_search

    assert await lookup.music.get_item_by_name("beatls", media_type=MediaType.ARTIST) is artist
    prompt = plugin.ai_query.await_args.args[0]
    assert 'one of "artist", or null' in prompt


def test_parse_reply_variants() -> None:
    """The reply parser accepts a fenced JSON object and drops what it cannot use."""
    allowed = (MediaType.TRACK, MediaType.ALBUM)
    fenced = '```json\n{"media_type": "Album", "name": " Abbey  Road ", "artist": "null"}\n```'
    assert parse_ai_lookup_response(fenced, allowed) == AILookupTerms(
        name="Abbey Road", media_type=MediaType.ALBUM
    )
    assert parse_ai_lookup_response('{"media_type": "playlist", "name": "x"}', allowed) == (
        AILookupTerms(name="x")
    )
    assert parse_ai_lookup_response('{"name": 5}', allowed) is None
    assert parse_ai_lookup_response("x" * 5000, allowed) is None
    assert parse_ai_lookup_response(42, allowed) is None


def test_prompt_rejects_oversized_or_empty_requests() -> None:
    """Only a short, non-empty request is sent to an engine."""
    assert build_ai_lookup_prompt("  ", None, None, (MediaType.TRACK,)) is None
    assert build_ai_lookup_prompt("a" * 600, None, None, (MediaType.TRACK,)) is None
    assert build_ai_lookup_prompt("Yesterday", None, None, (MediaType.TRACK,)) is not None

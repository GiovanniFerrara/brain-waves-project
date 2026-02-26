"""EventBus — synchronous pub/sub for EEG events."""

from __future__ import annotations

from collections.abc import Callable

from .base import Event, EventType

EventHandler = Callable[[Event], None]


class EventBus:
    """Simple synchronous event dispatcher.

    Usage::

        bus = EventBus()
        bus.subscribe(EventType.BLINK, lambda e: print(e))
        bus.publish(Event(EventType.BLINK, timestamp=1.0, value=300.0))
    """

    def __init__(self) -> None:
        self._handlers: dict[EventType | None, list[EventHandler]] = {}

    def subscribe(
        self,
        event_type: EventType | None,
        handler: EventHandler,
    ) -> None:
        """Subscribe to events. Pass ``None`` as type to receive all events."""
        self._handlers.setdefault(event_type, []).append(handler)

    def publish(self, event: Event) -> None:
        """Dispatch an event to matching handlers synchronously."""
        for handler in self._handlers.get(event.type, []):
            handler(event)
        # Also dispatch to wildcard subscribers
        for handler in self._handlers.get(None, []):
            handler(event)

#!/usr/bin/env python3
"""Main entry point — live EEG-to-sound."""

import asyncio
import signal

from thebox.config import TheBoxConfig
from thebox.pipeline import Pipeline


async def main():
    config = TheBoxConfig()
    pipeline = Pipeline(config)

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, lambda: asyncio.ensure_future(pipeline.stop()))

    await pipeline.start()


if __name__ == "__main__":
    asyncio.run(main())

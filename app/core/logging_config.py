import logging

from app.core.config import settings


def configure_logging() -> None:
    """Configure the root logger so every module's logger.info()/.warning() call
    actually emits output, regardless of how the process is started.

    Without this, the app relied on uvicorn's default logging setup — INFO logs
    from rag_engine, chat_service, etc. would silently disappear under any other
    ASGI server, in a script, or in a test that imports the app directly.
    """
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # basicConfig sets the level for every logger in the process, including
    # third-party ones. httpx logs an INFO line per HTTP request (noisy — the
    # gateway client and Hugging Face Hub downloads both use it under the hood),
    # so keep it at WARNING regardless of the app's own LOG_LEVEL.
    logging.getLogger("httpx").setLevel(logging.WARNING)

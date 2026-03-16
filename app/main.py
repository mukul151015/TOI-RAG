from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import get_settings
from app.db.database import close_pool, ensure_runtime_schema, ensure_schema, open_pool


settings = get_settings()
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("App startup: opening database pool")
    open_pool()
    if settings.db_auto_ensure_runtime_schema:
        logger.info("App startup: ensuring lightweight runtime schema")
        ensure_runtime_schema()
    else:
        logger.info("App startup: runtime schema ensure disabled by configuration")
    if settings.db_auto_ensure_schema:
        logger.info("App startup: ensuring schema")
        ensure_schema()
    else:
        logger.info("App startup: schema ensure disabled by configuration")
    logger.info("App startup complete")
    yield
    logger.info("App shutdown: closing database pool")
    close_pool()


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.include_router(router)

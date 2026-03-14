from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import get_settings
from app.db.database import close_pool, ensure_schema, open_pool


settings = get_settings()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    open_pool()
    ensure_schema()
    yield
    close_pool()


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.include_router(router)

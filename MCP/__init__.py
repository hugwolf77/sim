# weather_service/src/weather_service/__init__.py에 추가
from . import MCPserver
import asyncio

def main():
    """패키지의 메인 진입점."""
    asyncio.run(MCPserver.main())
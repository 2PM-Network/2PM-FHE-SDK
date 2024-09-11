from typing import Any, Callable, Generator, List

from pydantic import BaseModel

DEFAULT_SEGMENT_MAX_CHUNKS = 1024
DEFAULT_CHUNK_SIZE = 256
DEFAULT_SEGMENT_SIZE = DEFAULT_CHUNK_SIZE * DEFAULT_SEGMENT_MAX_CHUNKS


class H256(BaseModel):
    hash: str

    @classmethod
    def __get_validators__(cls) -> Generator[Callable[..., Any], None, None]:
        yield cls.validate

    @staticmethod
    def validate(value):
        if len(value.hash) != 66:
            raise ValueError("H256 must be a 64 hex digit string")
        return value


class Status(BaseModel):
    connected_peers: int


class NHProvider:
    def __init__(self, url: str):
        self.url = url
        self.client = AsyncClient(base_url=url)

    def request(self, method: str, params: List[Any]) -> Any:
        return self.client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
                "id": 1,
            },
        )

    async def get_status(self) -> Status:
        res = await self.request("zgs_getStatus", [])
        return Status(connected_peers=res.json()["result"]["connectedPeers"])

    async def upload_segment(self, seg: dict):
        res = await self.request("zgs_uploadSegment", [seg])
        return res.json()

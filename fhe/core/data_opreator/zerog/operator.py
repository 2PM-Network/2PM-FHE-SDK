import asyncio
from copy import copy

from .contract import flow_contract
from .provider import NHProvider
from .spec import TX_PARAMS
from .submission import create_submission, data_to_segments


class ZeroGStorageClient:
    def __init__(self, url):
        self.provider = NHProvider(url)

    async def upload_file(self, data: bytes):
        segments = data_to_segments(data)
        for segment in segments:
            await self.provider.upload_segment(segment)
        return True


class ZeroGOperatorMixin:
    @property
    def url(self):
        return self._url

    @url.setter
    def url(self, value):
        self._url = value
        self.storage_client = ZeroGStorageClient(value)

    async def register_file(self, file_path):
        file_path = ".".join(file_path.split(".")[:-1]) + ".zip"
        with open(file_path, "rb") as f:
            data = f.read()

        submissions, data_root = create_submission(data)
        try:
            tx_receipt = flow_contract.submit(submissions)
            print(f"Transaction receipt: {tx_receipt}")
        except Exception as e:
            print(f"Failed to submit: {e}")
            pass

        await self.storage_client.upload_file(data)

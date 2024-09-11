import os

from web3 import Web3

ENTRY_SIZE = 256
PORA_CHUNK_SIZE = 1024
PRIVATE_KEY = os.environ.get("PRIVATE_KEY")
GENESIS_ACCOUNT = Web3().eth.account.from_key(PRIVATE_KEY)
TX_PARAMS = {
    "gasPrice": 10_000_000_000,
    "gas": 10_000_000,
    "from": GENESIS_ACCOUNT.address,
}
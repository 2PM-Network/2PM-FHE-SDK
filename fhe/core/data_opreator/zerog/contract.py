import json
import os

from web3 import Web3

PRIVATE_KEY = os.getenv("PRIVATE_KEY")


class ContractConnector:
    def __init__(self, contract_address, contract_abi, rpc_url):
        self.web3 = Web3(Web3.HTTPProvider(rpc_url))
        if not self.web3.is_connected():
            raise Exception("Failed to connect to the 0g network")
        self.contract = self.web3.eth.contract(
            address=contract_address, abi=contract_abi
        )
        self.account = self.web3.eth.account.from_key(PRIVATE_KEY)

    def submit(self, submission):
        transaction = self.contract.functions.submit(submission).build_transaction(
            {
                "from": self.account.address,
                "nonce": self.web3.eth.get_transaction_count(self.account.address),
                "gas": 2000000,
                "gasPrice": self.web3.to_wei("20", "gwei"),
            }
        )

        # Sign the transaction
        signed_tx = self.web3.eth.account.sign_transaction(transaction, PRIVATE_KEY)

        # Send the transaction
        tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)

        # Get the transaction receipt
        tx_receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
        return tx_receipt


# Contract address and ABI (Application Binary Interface)
contract_address = "0xb8F03061969da6Ad38f0a4a9f8a86bE71dA3c8E7"

# Replace with your contract ABI
contract_abi = json.loads(
    """[
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "book_",
          "type": "address"
        },
        {
          "internalType": "uint256",
          "name": "blocksPerEpoch_",
          "type": "uint256"
        },
        {
          "internalType": "uint256",
          "name": "deployDelay_",
          "type": "uint256"
        }
      ],
      "stateMutability": "nonpayable",
      "type": "constructor"
    },
    {
      "inputs": [],
      "name": "InvalidSubmission",
      "type": "error"
    },
    {
      "anonymous": false,
      "inputs": [
        {
          "indexed": true,
          "internalType": "address",
          "name": "sender",
          "type": "address"
        },
        {
          "indexed": true,
          "internalType": "uint256",
          "name": "index",
          "type": "uint256"
        },
        {
          "indexed": false,
          "internalType": "bytes32",
          "name": "startMerkleRoot",
          "type": "bytes32"
        },
        {
          "indexed": false,
          "internalType": "uint256",
          "name": "submissionIndex",
          "type": "uint256"
        },
        {
          "indexed": false,
          "internalType": "uint256",
          "name": "flowLength",
          "type": "uint256"
        },
        {
          "indexed": false,
          "internalType": "bytes32",
          "name": "context",
          "type": "bytes32"
        }
      ],
      "name": "NewEpoch",
      "type": "event"
    },
    {
      "anonymous": false,
      "inputs": [
        {
          "indexed": false,
          "internalType": "address",
          "name": "account",
          "type": "address"
        }
      ],
      "name": "Paused",
      "type": "event"
    },
    {
      "anonymous": false,
      "inputs": [
        {
          "indexed": true,
          "internalType": "address",
          "name": "sender",
          "type": "address"
        },
        {
          "indexed": true,
          "internalType": "bytes32",
          "name": "identity",
          "type": "bytes32"
        },
        {
          "indexed": false,
          "internalType": "uint256",
          "name": "submissionIndex",
          "type": "uint256"
        },
        {
          "indexed": false,
          "internalType": "uint256",
          "name": "startPos",
          "type": "uint256"
        },
        {
          "indexed": false,
          "internalType": "uint256",
          "name": "length",
          "type": "uint256"
        },
        {
          "components": [
            {
              "internalType": "uint256",
              "name": "length",
              "type": "uint256"
            },
            {
              "internalType": "bytes",
              "name": "tags",
              "type": "bytes"
            },
            {
              "components": [
                {
                  "internalType": "bytes32",
                  "name": "root",
                  "type": "bytes32"
                },
                {
                  "internalType": "uint256",
                  "name": "height",
                  "type": "uint256"
                }
              ],
              "internalType": "struct SubmissionNode[]",
              "name": "nodes",
              "type": "tuple[]"
            }
          ],
          "indexed": false,
          "internalType": "struct Submission",
          "name": "submission",
          "type": "tuple"
        }
      ],
      "name": "Submit",
      "type": "event"
    },
    {
      "anonymous": false,
      "inputs": [
        {
          "indexed": false,
          "internalType": "address",
          "name": "account",
          "type": "address"
        }
      ],
      "name": "Unpaused",
      "type": "event"
    },
    {
      "inputs": [
        {
          "components": [
            {
              "internalType": "uint256",
              "name": "length",
              "type": "uint256"
            },
            {
              "internalType": "bytes",
              "name": "tags",
              "type": "bytes"
            },
            {
              "components": [
                {
                  "internalType": "bytes32",
                  "name": "root",
                  "type": "bytes32"
                },
                {
                  "internalType": "uint256",
                  "name": "height",
                  "type": "uint256"
                }
              ],
              "internalType": "struct SubmissionNode[]",
              "name": "nodes",
              "type": "tuple[]"
            }
          ],
          "internalType": "struct Submission[]",
          "name": "submissions",
          "type": "tuple[]"
        }
      ],
      "name": "batchSubmit",
      "outputs": [
        {
          "internalType": "uint256[]",
          "name": "indexes",
          "type": "uint256[]"
        },
        {
          "internalType": "bytes32[]",
          "name": "digests",
          "type": "bytes32[]"
        },
        {
          "internalType": "uint256[]",
          "name": "startIndexes",
          "type": "uint256[]"
        },
        {
          "internalType": "uint256[]",
          "name": "lengths",
          "type": "uint256[]"
        }
      ],
      "stateMutability": "payable",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "blocksPerEpoch",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "book",
      "outputs": [
        {
          "internalType": "contract AddressBook",
          "name": "",
          "type": "address"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "commitRoot",
      "outputs": [],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "currentLength",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "epoch",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "epochStartPosition",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "firstBlock",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "getContext",
      "outputs": [
        {
          "components": [
            {
              "internalType": "uint256",
              "name": "epoch",
              "type": "uint256"
            },
            {
              "internalType": "uint256",
              "name": "mineStart",
              "type": "uint256"
            },
            {
              "internalType": "bytes32",
              "name": "flowRoot",
              "type": "bytes32"
            },
            {
              "internalType": "uint256",
              "name": "flowLength",
              "type": "uint256"
            },
            {
              "internalType": "bytes32",
              "name": "blockDigest",
              "type": "bytes32"
            },
            {
              "internalType": "bytes32",
              "name": "digest",
              "type": "bytes32"
            }
          ],
          "internalType": "struct MineContext",
          "name": "",
          "type": "tuple"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "bytes32",
          "name": "digest",
          "type": "bytes32"
        }
      ],
      "name": "getEpochRange",
      "outputs": [
        {
          "components": [
            {
              "internalType": "uint128",
              "name": "start",
              "type": "uint128"
            },
            {
              "internalType": "uint128",
              "name": "end",
              "type": "uint128"
            }
          ],
          "internalType": "struct EpochRange",
          "name": "",
          "type": "tuple"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "makeContext",
      "outputs": [],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "uint256",
          "name": "cnt",
          "type": "uint256"
        }
      ],
      "name": "makeContextFixedTimes",
      "outputs": [],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "makeContextWithResult",
      "outputs": [
        {
          "components": [
            {
              "internalType": "uint256",
              "name": "epoch",
              "type": "uint256"
            },
            {
              "internalType": "uint256",
              "name": "mineStart",
              "type": "uint256"
            },
            {
              "internalType": "bytes32",
              "name": "flowRoot",
              "type": "bytes32"
            },
            {
              "internalType": "uint256",
              "name": "flowLength",
              "type": "uint256"
            },
            {
              "internalType": "bytes32",
              "name": "blockDigest",
              "type": "bytes32"
            },
            {
              "internalType": "bytes32",
              "name": "digest",
              "type": "bytes32"
            }
          ],
          "internalType": "struct MineContext",
          "name": "",
          "type": "tuple"
        }
      ],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "uint256",
          "name": "_length",
          "type": "uint256"
        },
        {
          "internalType": "uint256",
          "name": "alignExp",
          "type": "uint256"
        }
      ],
      "name": "nextAlign",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "",
          "type": "uint256"
        }
      ],
      "stateMutability": "pure",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "uint256",
          "name": "_length",
          "type": "uint256"
        }
      ],
      "name": "nextPow2",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "",
          "type": "uint256"
        }
      ],
      "stateMutability": "pure",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "numSubmissions",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "paused",
      "outputs": [
        {
          "internalType": "bool",
          "name": "",
          "type": "bool"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "uint128",
          "name": "targetPosition",
          "type": "uint128"
        }
      ],
      "name": "queryContextAtPosition",
      "outputs": [
        {
          "components": [
            {
              "internalType": "uint128",
              "name": "start",
              "type": "uint128"
            },
            {
              "internalType": "uint128",
              "name": "end",
              "type": "uint128"
            },
            {
              "internalType": "bytes32",
              "name": "digest",
              "type": "bytes32"
            }
          ],
          "internalType": "struct EpochRangeWithContextDigest",
          "name": "range",
          "type": "tuple"
        }
      ],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "root",
      "outputs": [
        {
          "internalType": "bytes32",
          "name": "",
          "type": "bytes32"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "rootHistory",
      "outputs": [
        {
          "internalType": "contract IDigestHistory",
          "name": "",
          "type": "address"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "submissionIndex",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "components": [
            {
              "internalType": "uint256",
              "name": "length",
              "type": "uint256"
            },
            {
              "internalType": "bytes",
              "name": "tags",
              "type": "bytes"
            },
            {
              "components": [
                {
                  "internalType": "bytes32",
                  "name": "root",
                  "type": "bytes32"
                },
                {
                  "internalType": "uint256",
                  "name": "height",
                  "type": "uint256"
                }
              ],
              "internalType": "struct SubmissionNode[]",
              "name": "nodes",
              "type": "tuple[]"
            }
          ],
          "internalType": "struct Submission",
          "name": "submission",
          "type": "tuple"
        }
      ],
      "name": "submit",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "",
          "type": "uint256"
        },
        {
          "internalType": "bytes32",
          "name": "",
          "type": "bytes32"
        },
        {
          "internalType": "uint256",
          "name": "",
          "type": "uint256"
        },
        {
          "internalType": "uint256",
          "name": "",
          "type": "uint256"
        }
      ],
      "stateMutability": "payable",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "unstagedHeight",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "uint256",
          "name": "height",
          "type": "uint256"
        }
      ],
      "name": "zeros",
      "outputs": [
        {
          "internalType": "bytes32",
          "name": "",
          "type": "bytes32"
        }
      ],
      "stateMutability": "pure",
      "type": "function"
    }
  ]"""
)

flow_contract = ContractConnector(
    contract_address, contract_abi, "https://rpc-testnet.0g.ai"
)

from io import StringIO

from .core.data_opreator.zerog.operator import ZeroGOperatorMixin
from .utils.fhe import fhe_client
from .utils.file import convert_file_to_df


class FHEClient(ZeroGOperatorMixin):
    def __init__(self, url):
        self.url = url

    def fhe_encrypt(self, path: str, encrypted_path: str):
        """Encrypts a file using FHE.

        Args:
            file_name (str): The name of the file.
            content (StringIO): The content of the file.

        Returns:
            pd.DataFrame: The encrypted DataFrame.
        """
        df = convert_file_to_df(path)
        df_encrypted = fhe_client.encrypt_from_pandas(df)
        df_encrypted.save(encrypted_path)

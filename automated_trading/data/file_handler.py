import os


class LocalFileHandler:
    """
    本地文件操作。
    """

    @staticmethod
    def exists(filepath: str) -> bool:
        return os.path.exists(filepath)

    @staticmethod
    def read(filepath: str) -> str:
        with open(filepath, 'r') as f:
            return f.read()

    @staticmethod
    def write(filepath: str, content: str):
        with open(filepath, 'w') as f:
            f.write(content)

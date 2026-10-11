
import logging
import os
import sys

class LoggingService:
    _initialized = False

    @classmethod
    def setup(cls, log_level: str = "INFO", log_dir: str = "logs"):
        """全局初始化一次即可（幂等操作）"""
        if cls._initialized:
            return

        # 锚定到绝对路径，防止日志到处乱拉
        base_dir = os.getcwd()
        target_dir = os.path.join(base_dir, log_dir)
        os.makedirs(target_dir, exist_ok=True)
        log_file = os.path.join(target_dir, "agent.Log")

        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"  # 注意加入了 [%(name)s]
        )
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)

        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

        cls._initialized = True
"""
统一日志管理
"""
import logging
import os
from config import cfg


def setup_logger():
    logger = logging.getLogger("CTSO_Logger")
    logger.setLevel(logging.INFO)

    # 防止重复添加 handler
    if not logger.handlers:
        # 控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)

        logger.addHandler(ch)

        # 文件处理器 (可选，记录详细日志)
        log_file = os.path.join(cfg.OUTPUT_DIR, "app.log")
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


log = setup_logger()
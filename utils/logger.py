import logging
from logging.handlers import RotatingFileHandler

# Create a global logger instance
logger = logging.getLogger("logger")
logger.setLevel(logging.DEBUG)

# Check if the logger already has handlers
if not logger.hasHandlers():
    # Log dosyası ayarları
    file_handler = RotatingFileHandler("logs/predictive-maintenance.log", maxBytes=100000000, backupCount=10, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    # Log formatı
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    # Handler'ı logger'a ekleme
    logger.addHandler(file_handler)
    
    
    
    
info_logger = logging.getLogger("info_logger")
info_logger.setLevel(logging.DEBUG)

if info_logger.hasHandlers():
    info_logger.handlers.clear()



if not info_logger.hasHandlers():
    handler = logging.FileHandler("logs/info-log.log")
    handler.setLevel(logging.DEBUG)  # Important
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    info_logger.addHandler(handler)


info_logger = logging.getLogger("info_logger")
info_logger.setLevel(logging.DEBUG)

if not info_logger.hasHandlers():
    handler = logging.FileHandler("logs/info-log.log")
    formating = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formating)

    info_logger.addHandler(handler)

feature_logger = logging.getLogger("feature_logger")
feature_logger.setLevel(logging.DEBUG)

if not feature_logger.hasHandlers():
    handler = logging.FileHandler("logs/feature.log")
    formating = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formating)
    feature_logger.addHandler(handler)


ph3_logger = logging.getLogger("phase3_logger")
ph3_logger.setLevel(logging.DEBUG)

if not ph3_logger.hasHandlers():
    handler = logging.FileHandler("logs/phase3_logger.log")
    formating = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formating)
    ph3_logger.addHandler(handler)


ph3_1_logger = logging.getLogger("phase3_1_logger")
ph3_1_logger.setLevel(logging.DEBUG)
if not ph3_1_logger.hasHandlers():
    handler = RotatingFileHandler("logs/phase3_1_logger.log", maxBytes=100000000, backupCount=10, encoding='utf-8')
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    ph3_1_logger.addHandler(handler)


logger_agg = logging.getLogger("logger_agg")
logger_agg.setLevel(logging.DEBUG)

# Check if the logger already has handlers
if not logger_agg.hasHandlers():
    # Log dosyası ayarları
    file_handler = RotatingFileHandler("logs/aggregation-log.log", maxBytes=100000000, backupCount=10, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    # Log formatı
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    # Handler'ı logger'a ekleme
    logger_agg.addHandler(file_handler)


ph1_logger = logging.getLogger("phase1_logger")
ph1_logger.setLevel(logging.DEBUG)

if not ph1_logger.hasHandlers():
    handler = RotatingFileHandler("logs/phase1_logger.log", maxBytes=100000000, backupCount=10, encoding='utf-8')
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    ph1_logger.addHandler(handler)

ph2_logger = logging.getLogger("phase2_logger")
ph2_logger.setLevel(logging.DEBUG)

if not ph2_logger.hasHandlers():
    handler = RotatingFileHandler("logs/phase2_logger.log", maxBytes=100000000, backupCount=10, encoding='utf-8')
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    ph2_logger.addHandler(handler)
    
import logging
from logging.handlers import RotatingFileHandler

# Create a global logger instance
logger = logging.getLogger("logger")
logger.setLevel(logging.DEBUG)

# Check if the logger already has handlers
if not logger.hasHandlers():
    # Log dosyası ayarları
    file_handler = RotatingFileHandler("logs/predictive-maintenance.log", maxBytes=100000000, backupCount=10, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    # Log formatı
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    # Handler'ı logger'a ekleme
    logger.addHandler(file_handler)

info_logger = logging.getLogger("info_logger")
info_logger.setLevel(logging.DEBUG)

if info_logger.hasHandlers():
    info_logger.handlers.clear()

if not info_logger.hasHandlers():
    handler = logging.FileHandler("logs/info-log.log")
    handler.setLevel(logging.DEBUG)  # Important
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    info_logger.addHandler(handler)

feature_logger = logging.getLogger("feature_logger")
feature_logger.setLevel(logging.DEBUG)

if not feature_logger.hasHandlers():
    handler = logging.FileHandler("logs/feature.log")
    formating = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formating)
    feature_logger.addHandler(handler)

ph3_logger = logging.getLogger("phase3_logger")
ph3_logger.setLevel(logging.DEBUG)

if not ph3_logger.hasHandlers():
    handler = logging.FileHandler("logs/phase3_logger.log")
    formating = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formating)
    ph3_logger.addHandler(handler)

ph3_1_logger = logging.getLogger("phase3_1_logger")
ph3_1_logger.setLevel(logging.DEBUG)
if not ph3_1_logger.hasHandlers():
    handler = RotatingFileHandler("logs/phase3_1_logger.log", maxBytes=100000000, backupCount=10, encoding='utf-8')
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    ph3_1_logger.addHandler(handler)

logger_agg = logging.getLogger("logger_agg")
logger_agg.setLevel(logging.DEBUG)

# Check if the logger already has handlers
if not logger_agg.hasHandlers():
    # Log dosyası ayarları
    file_handler = RotatingFileHandler("logs/aggregation-log.log", maxBytes=100000000, backupCount=10, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    # Log formatı
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    # Handler'ı logger'a ekleme
    logger_agg.addHandler(file_handler)

ph1_logger = logging.getLogger("phase1_logger")
ph1_logger.setLevel(logging.DEBUG)

if not ph1_logger.hasHandlers():
    handler = RotatingFileHandler("logs/phase1_logger.log", maxBytes=100000000, backupCount=10, encoding='utf-8')
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    ph1_logger.addHandler(handler)

ph2_logger = logging.getLogger("phase2_logger")
ph2_logger.setLevel(logging.DEBUG)

if not ph2_logger.hasHandlers():
    handler = RotatingFileHandler("logs/phase2_logger.log", maxBytes=100000000, backupCount=10, encoding='utf-8')
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    ph2_logger.addHandler(handler)




# import logging
# from logging.handlers import RotatingFileHandler

# # Create a global logger instance
# logger = logging.getLogger("logger")
# logger.setLevel(logging.INFO)

# if not logger.hasHandlers():
#     file_handler = RotatingFileHandler("logs/predictive-maintenance.log", maxBytes=100000000, backupCount=10, encoding='utf-8')
#     file_handler.setLevel(logging.DEBUG)
#     formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
#     file_handler.setFormatter(formatter)
#     logger.addHandler(file_handler)

# info_logger = logging.getLogger("info_logger")
# info_logger.setLevel(logging.INFO)

# if info_logger.hasHandlers():
#     info_logger.handlers.clear()

# if not info_logger.hasHandlers():
#     handler = RotatingFileHandler("logs/info-log.log", maxBytes=100000000, backupCount=10, encoding='utf-8')
#     handler.setLevel(logging.DEBUG)
#     formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
#     handler.setFormatter(formatter)
#     info_logger.addHandler(handler)

# feature_logger = logging.getLogger("feature_logger")
# feature_logger.setLevel(logging.INFO)

# if not feature_logger.hasHandlers():
#     handler = RotatingFileHandler("logs/feature.log", maxBytes=100000000, backupCount=10, encoding='utf-8')
#     handler.setLevel(logging.DEBUG)
#     formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
#     handler.setFormatter(formatter)
#     feature_logger.addHandler(handler)

# ph3_logger = logging.getLogger("phase3_logger")
# ph3_logger.setLevel(logging.INFO)

# if not ph3_logger.hasHandlers():
#     handler = RotatingFileHandler("logs/phase3_logger.log", maxBytes=100000000, backupCount=10, encoding='utf-8')
#     handler.setLevel(logging.DEBUG)
#     formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
#     handler.setFormatter(formatter)
#     ph3_logger.addHandler(handler)

# ph3_1_logger = logging.getLogger("phase3_1_logger")
# ph3_1_logger.setLevel(logging.INFO)

# if not ph3_1_logger.hasHandlers():
#     handler = RotatingFileHandler("logs/phase3_1_logger.log", maxBytes=100000000, backupCount=10, encoding='utf-8')
#     handler.setLevel(logging.DEBUG)
#     formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
#     handler.setFormatter(formatter)
#     ph3_1_logger.addHandler(handler)

# logger_agg = logging.getLogger("logger_agg")
# logger_agg.setLevel(logging.INFO)

# if not logger_agg.hasHandlers():
#     file_handler = RotatingFileHandler("logs/aggregation-log.log", maxBytes=100000000, backupCount=10, encoding='utf-8')
#     file_handler.setLevel(logging.DEBUG)
#     formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
#     file_handler.setFormatter(formatter)
#     logger_agg.addHandler(file_handler)

# ph1_logger = logging.getLogger("phase1_logger")
# ph1_logger.setLevel(logging.INFO)

# if not ph1_logger.hasHandlers():
#     handler = RotatingFileHandler("logs/phase1_logger.log", maxBytes=100000000, backupCount=10, encoding='utf-8')
#     handler.setLevel(logging.DEBUG)
#     formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
#     handler.setFormatter(formatter)
#     ph1_logger.addHandler(handler)

# ph2_logger = logging.getLogger("phase2_logger")
# ph2_logger.setLevel(logging.INFO)

# if not ph2_logger.hasHandlers():
#     handler = RotatingFileHandler("logs/phase2_logger.log", maxBytes=100000000, backupCount=10, encoding='utf-8')
#     handler.setLevel(logging.DEBUG)
#     formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
#     handler.setFormatter(formatter)
#     ph2_logger.addHandler(handler)
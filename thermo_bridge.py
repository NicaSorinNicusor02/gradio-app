import os, sys, types, logging

# Root next to app.py
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ---- simple_logger shim (the repo imports this)
_simple_logger_mod = types.ModuleType("simple_logger")

def _to_level(val):
    if isinstance(val, int): return val
    try:
        name = getattr(val, "name", val)
        return getattr(logging, str(name).upper())
    except Exception:
        return logging.INFO

class _LogLevel:
    DEBUG   = logging.DEBUG
    INFO    = logging.INFO
    WARNING = logging.WARNING
    ERROR   = logging.ERROR
    CRITICAL= logging.CRITICAL

class _Logger:
    _console_level = logging.INFO
    _file_level    = logging.DEBUG
    _log_file      = None
    _logger        = None
    @classmethod
    def set_console_logging_level(cls, level): cls._console_level = _to_level(level)
    @classmethod
    def set_file_logging_level(cls, level):    cls._file_level    = _to_level(level)
    @classmethod
    def set_log_file(cls, filename):           cls._log_file      = filename
    @classmethod
    def init(cls):
        log = logging.getLogger("thermography")
        log.setLevel(logging.DEBUG)
        log.handlers.clear()
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        sh = logging.StreamHandler(sys.stdout); sh.setLevel(cls._console_level); sh.setFormatter(fmt)
        log.addHandler(sh)
        if cls._log_file:
            fh = logging.FileHandler(cls._log_file); fh.setLevel(cls._file_level); fh.setFormatter(fmt)
            log.addHandler(fh)
        cls._logger = log
    @classmethod
    def _get(cls): return cls._logger or logging.getLogger("thermography")
    @classmethod
    def info(cls,*a,**k): cls._get().info(*a,**k)
    @classmethod
    def debug(cls,*a,**k): cls._get().debug(*a,**k)
    @classmethod
    def warning(cls,*a,**k): cls._get().warning(*a,**k)
    @classmethod
    def error(cls,*a,**k): cls._get().error(*a,**k)
    @classmethod
    def critical(cls,*a,**k): cls._get().critical(*a,**k)

_simple_logger_mod.Logger = _Logger
_simple_logger_mod.LogLevel = _LogLevel
sys.modules["simple_logger"] = _simple_logger_mod

# ---- import repo classes
THERMO_OK = True
THERMO_IMPORT_ERR = ""
try:
    TH_DIR = os.path.join(ROOT, "thermography")
    if os.path.isdir(TH_DIR) and TH_DIR not in sys.path:
        sys.path.insert(0, TH_DIR)

    from thermography.detection.preprocessing import FramePreprocessor, PreprocessingParams# type: ignore 
    from thermography.detection.edge_detection import EdgeDetector, EdgeDetectorParams # type: ignore 
    from thermography.detection.segment_detection import SegmentDetector, SegmentDetectorParams # type: ignore
    from thermography.detection.segment_clustering import SegmentClusterer, SegmentClustererParams, ClusterCleaningParams # type: ignore
    from thermography.detection.intersection_detection import IntersectionDetector, IntersectionDetectorParams # type: ignore
    from thermography.detection.rectangle_detection import RectangleDetector, RectangleDetectorParams # type: ignore

    from types import SimpleNamespace
    REPO = SimpleNamespace(
        FramePreprocessor=FramePreprocessor, PreprocessingParams=PreprocessingParams,
        EdgeDetector=EdgeDetector, EdgeDetectorParams=EdgeDetectorParams,
        SegmentDetector=SegmentDetector, SegmentDetectorParams=SegmentDetectorParams,
        SegmentClusterer=SegmentClusterer, SegmentClustererParams=SegmentClustererParams, ClusterCleaningParams=ClusterCleaningParams,
        IntersectionDetector=IntersectionDetector, IntersectionDetectorParams=IntersectionDetectorParams,
        RectangleDetector=RectangleDetector, RectangleDetectorParams=RectangleDetectorParams
    )
except Exception as _e:
    THERMO_OK = False
    THERMO_IMPORT_ERR = f"{type(_e).__name__}: {_e}"
    REPO = None

def repo():
    if not THERMO_OK or REPO is None:
        raise ImportError(THERMO_IMPORT_ERR)
    return REPO

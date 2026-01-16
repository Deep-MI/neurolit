import logging
import sys

def get_logger(name: str = "LIT") -> logging.Logger:
    """
    Get a logger with a default configuration if not already configured.
    
    Parameters
    ----------
    name : str
        The name of the logger.
        
    Returns
    -------
    logging.Logger
        The configured logger.
    """
    logger = logging.getLogger(name)
    
    # Only add handlers if the logger doesn't have any yet
    # and it's not inheriting from a logger that has them
    if not logger.handlers and not logger.parent.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        # Standard FastSurfer-like format could be used here
        formatter = logging.Formatter(
            "[%(levelname)s] %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        # Prevent propagation to root logger if we've added our own handler
        logger.propagate = False
        
    return logger

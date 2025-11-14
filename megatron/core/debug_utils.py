DEBUG = False

def is_debug_enabled():
    """Check if debug mode is enabled.
    
    Returns True if --debug flag was set, False otherwise.
    This function can be called even before args are parsed (returns False in that case).
    """
    return DEBUG
    # try:
    #     from megatron.training.global_vars import get_args
    #     args = get_args()
    #     return getattr(args, 'debug', False)
    # except (AttributeError, RuntimeError, AssertionError):
    #     # Args not initialized yet, return the global DEBUG flag
    #     return DEBUG

def debug_log(logger, level, message, *args, **kwargs):
    """Conditionally log a message only if debug mode is enabled.
    
    Args:
        logger: The logging.Logger instance
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
        message: The message to log
        *args, **kwargs: Additional arguments passed to logger.log()
    """
    if is_debug_enabled():
        logger.log(level, message, *args, **kwargs)

def debug_assert(condition, message=None):
    """Conditionally assert only if debug mode is enabled.
    
    Args:
        condition: The condition to check
        message: Optional error message if assertion fails
    """
    if is_debug_enabled():
        if message is not None:
            assert condition, message
        else:
            assert condition

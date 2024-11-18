import logging
import sys

def configure_logging():
    """
    Configures the logging settings to ensure logs appear in the terminal.
    """
    logging.basicConfig(
        level=logging.DEBUG,  # Set to DEBUG to see all logs
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)  # Force logs to terminal (stdout)
        ]
    )
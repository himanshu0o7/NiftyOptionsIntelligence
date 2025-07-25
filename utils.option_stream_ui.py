# If the file is in a subdirectory like 'utils'
from utils.option_stream_ui import get_option_data

# OR if you need to add the directory to the path:
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from option_stream_ui import get_option_data


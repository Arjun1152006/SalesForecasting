import logging
import sys
import traceback

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('cmdstanpy').setLevel(logging.DEBUG)
logging.getLogger('prophet').setLevel(logging.DEBUG)

try:
    from prophet import Prophet
    print("Prophet loaded.")
    p = Prophet()
    print("Prophet initialized successfully!")
except Exception as e:
    print("Failed to initialize Prophet:")
    traceback.print_exc()

    # Let's inspect cmdstanpy and standard package paths
    try:
        import cmdstanpy
        print("cmdstanpy version:", cmdstanpy.__version__)
        print("cmdstanpy path:", cmdstanpy.__file__)
        print("cmdstan path:", cmdstanpy.utils.cmdstan_path())
    except Exception as e2:
        print("Failed to inspect cmdstanpy:")
        traceback.print_exc()

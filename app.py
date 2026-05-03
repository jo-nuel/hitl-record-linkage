from pathlib import Path
import runpy


if __name__ == "__main__":
    # Keep a simple root entrypoint so the documented command is
    # `streamlit run app.py` while the main app code stays in app/.
    runpy.run_path(str(Path(__file__).parent / "app" / "streamlit_app.py"), run_name="__main__")
